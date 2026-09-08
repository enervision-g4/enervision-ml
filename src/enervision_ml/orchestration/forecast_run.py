"""Assemble extraction, entrainement et ecriture pour la commande forecast.

Un commit par site, apres ses previsions et ses recommandations : un site en echec est
journalise, annule, et compte dans sites_failed sans arreter la boucle sur le reste du
parc, meme regle que realtime_collector.py cote ETL.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Optional, cast

from ..extract.history_source import HistorySourceLike
from ..load.prediction_repository import fetch_latest_for_target
from ..load.prediction_repository import insert_many as insert_predictions
from ..load.recommendation_repository import insert_many as insert_recommendations
from ..logging_setup import get_logger
from ..model.forecaster import ConsumptionForecaster
from ..postgres_connection import ConnectionLike
from ..records import Observation, PredictionRow
from ..transform.forecast_accuracy import (
    compute_forecast_accuracy,
    most_recently_resolved_observation,
)
from ..transform.prediction_drafts import build_prediction_rows, build_target_timestamps
from ..transform.recommendations import build_recommendations
from ..transform.thresholds import compute_threshold_kw
from ..transform.weather_outlook import Climatology, build_climatology, project_weather
from .experiment_tracking import ExperimentTrackingLogger

logger = get_logger("forecast_run")


@dataclass(frozen=True)
class ForecastReport:
    """Resultat d'un lot de prevision.

    Attributes:
        sites_forecast: Sites pour lesquels au moins une prevision a ete ecrite.
        sites_skipped: Sites ignores faute d'historique suffisant.
        sites_failed: Sites dont l'ecriture a echoue et a ete annulee.
        recommendations_written: Nombre total de recommandations ecrites sur le lot.
    """

    sites_forecast: list[str] = field(default_factory=list)
    sites_skipped: list[str] = field(default_factory=list)
    sites_failed: list[str] = field(default_factory=list)
    recommendations_written: int = 0


class ForecastRun:
    """Orchestre un lot de prevision : un modele entraine puis ecrit, par site."""

    def __init__(
        self,
        history_source: HistorySourceLike,
        connection: ConnectionLike,
        horizon_hours: int,
        minimum_training_hours: int,
        threshold_ratio: float,
        create_forecaster: Callable[[str], ConsumptionForecaster] = ConsumptionForecaster,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        experiment_logger: Optional[ExperimentTrackingLogger] = None,
    ) -> None:
        """Prepare le run.

        Args:
            history_source: Source d'historique, CSV ou base.
            connection: Connexion ouverte vers la base de destination.
            horizon_hours: Nombre d'heures de prevision ecrites par site.
            minimum_training_hours: Historique minimal exige pour entrainer un site.
            threshold_ratio: Fraction de capacity_kw retenue comme seuil d'alerte.
            create_forecaster: Fabrique du modele, injectee par les tests.
            now: Source de l'instant courant, injectee par les tests.
            experiment_logger: Journal MLflow de la justesse en production, ou None
                pour desactiver ce suivi (comportement par defaut, retro-compatible).
        """
        self._history_source = history_source
        self._connection = connection
        self._horizon_hours = horizon_hours
        self._minimum_training_hours = minimum_training_hours
        self._threshold_ratio = threshold_ratio
        self._create_forecaster = create_forecaster
        self._now = now
        self._experiment_logger = experiment_logger

    def run(self) -> ForecastReport:
        """Entraine et ecrit les previsions de chaque site du referentiel.

        Returns:
            Le rapport du lot, voir ForecastReport.
        """
        # ForecastReport est fige : on accumule dans des variables locales et on ne
        # construit l'instance qu'au retour, plutot que de reassigner un de ses champs
        # (recommendations_written) en cours de route, ce qu'un dataclass fige refuse.
        sites_forecast: list[str] = []
        sites_skipped: list[str] = []
        sites_failed: list[str] = []
        recommendations_written = 0
        generated_at = self._now()

        for site in self._history_source.load_site_catalog():
            observations = [
                observation
                for observation in self._history_source.load_observations(site.site_id)
                if observation.consumption_kw is not None
            ]
            self._grade_previous_forecast(site.site_id, observations)

            if len(observations) < self._minimum_training_hours:
                logger.warning(
                    "training_history_insufficient",
                    site_id=site.site_id,
                    available_hours=len(observations),
                    required_hours=self._minimum_training_hours,
                )
                sites_skipped.append(site.site_id)
                continue

            try:
                prediction_rows = self._forecast_one_site(
                    site.site_id, site.capacity_kw, observations, generated_at
                )
                recommendation_rows = build_recommendations(prediction_rows)

                insert_predictions(self._connection, prediction_rows)
                insert_recommendations(self._connection, recommendation_rows)
                self._connection.commit()

                sites_forecast.append(site.site_id)
                recommendations_written += len(recommendation_rows)
            except Exception:
                self._connection.rollback()
                logger.exception("forecast_failed", site_id=site.site_id)
                sites_failed.append(site.site_id)

        return ForecastReport(
            sites_forecast=sites_forecast,
            sites_skipped=sites_skipped,
            sites_failed=sites_failed,
            recommendations_written=recommendations_written,
        )

    def _grade_previous_forecast(self, site_id: str, observations: list[Observation]) -> None:
        """Confronte la derniere prevision resolue a la mesure reelle, si possible.

        Best-effort au meme titre que le suivi MLflow d'evaluate : ni une base
        injoignable en lecture ni un echec de journalisation ne doivent faire echouer
        le lot de prevision du site. Une lecture en echec annule la transaction en
        cours pour que l'ecriture des nouvelles previsions, plus bas, reparte d'une
        transaction saine.

        Args:
            site_id: Site concerne.
            observations: Historique horaire deja charge pour ce site.
        """
        if self._experiment_logger is None:
            return

        latest_observation = most_recently_resolved_observation(observations)
        if latest_observation is None or latest_observation.consumption_kw is None:
            return

        try:
            previous_prediction = fetch_latest_for_target(
                self._connection, site_id, latest_observation.timestamp
            )
        except Exception:
            self._connection.rollback()
            logger.exception("forecast_accuracy_lookup_failed", site_id=site_id)
            return

        if previous_prediction is None:
            return

        accuracy = compute_forecast_accuracy(
            previous_prediction.predicted_consumption_kw, latest_observation.consumption_kw
        )
        self._experiment_logger.log_forecast_accuracy(
            site_id=site_id,
            model_version=previous_prediction.model_version,
            target_timestamp=latest_observation.timestamp,
            accuracy=accuracy,
        )

    def _forecast_one_site(
        self,
        site_id: str,
        capacity_kw: Optional[int],
        observations: list[Observation],
        generated_at: datetime,
    ) -> list[PredictionRow]:
        forecaster = self._create_forecaster(site_id)
        forecaster.fit(observations)

        target_timestamps = build_target_timestamps(generated_at, self._horizon_hours)
        climatology = build_climatology(observations)
        future_observations = [
            self._build_future_observation(site_id, target_timestamp, climatology)
            for target_timestamp in target_timestamps
        ]
        predicted_consumption_kw = forecaster.predict(future_observations)

        observed_consumption_kw = [
            cast(float, observation.consumption_kw) for observation in observations
        ]
        threshold_kw = compute_threshold_kw(
            capacity_kw, observed_consumption_kw, self._threshold_ratio
        )

        return build_prediction_rows(
            site_id=site_id,
            target_timestamps=target_timestamps,
            predicted_consumption_kw=predicted_consumption_kw,
            threshold_kw=threshold_kw,
            model_version=forecaster.model_version,
            generated_at=generated_at,
        )

    @staticmethod
    def _build_future_observation(
        site_id: str, target_timestamp: datetime, climatology: Climatology
    ) -> Observation:
        # La temperature et l'humidite reelles de l'heure visee ne sont pas connues au
        # moment de predire : la moyenne climatologique (mois, heure) en tient lieu.
        weather = project_weather(climatology, target_timestamp)
        return Observation(
            site_id=site_id,
            timestamp=target_timestamp,
            consumption_kw=None,
            temperature_celsius=weather.temperature_celsius,
            humidity_percent=weather.humidity_percent,
        )
