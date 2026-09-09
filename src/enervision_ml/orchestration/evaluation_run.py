"""Assemble extraction, entrainement et evaluation pour la commande evaluate.

N'ecrit rien en base : sert uniquement a prouver, sans aucune infrastructure, que le
modele bat la baseline du profil horaire.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

from ..extract.history_source import HistorySourceLike
from ..model.forecaster import ConsumptionForecaster
from ..records import Observation
from ..transform.baseline import HourlyProfile, fit_hourly_profile, predict_from_profile
from ..transform.evaluation import SiteEvaluation, evaluate_site, split_chronologically

MINIMUM_OBSERVATIONS_FOR_EVALUATION = 48
"""En dessous, le split train/test n'a pas assez de matiere pour etre significatif."""


@dataclass(frozen=True)
class EvaluationReport:
    """Resultat d'un run d'evaluation.

    Attributes:
        evaluations: Comparaisons modele/baseline, une par site evalue.
        sites_skipped: Sites ignores faute d'historique suffisant.
    """

    evaluations: list[SiteEvaluation]
    sites_skipped: list[str]


class EvaluationRun:
    """Orchestre l'evaluation du modele contre la baseline, sans effet de bord."""

    def __init__(
        self,
        history_source: HistorySourceLike,
        test_ratio: float = 0.2,
        minimum_observations: int = MINIMUM_OBSERVATIONS_FOR_EVALUATION,
        create_forecaster: Callable[[str], ConsumptionForecaster] = ConsumptionForecaster,
    ) -> None:
        """Prepare le run.

        Args:
            history_source: Source d'historique, CSV ou base.
            test_ratio: Fraction de l'historique reservee au test.
            minimum_observations: Historique minimal exige pour evaluer un site.
            create_forecaster: Fabrique du modele, injectee par les tests pour eviter
                de dependre de scikit-learn.
        """
        self._history_source = history_source
        self._test_ratio = test_ratio
        self._minimum_observations = minimum_observations
        self._create_forecaster = create_forecaster

    def run(self) -> EvaluationReport:
        """Evalue chaque site du referentiel.

        Returns:
            Le rapport d'evaluation, avec les sites ignores le cas echeant.
        """
        evaluations: list[SiteEvaluation] = []
        sites_skipped: list[str] = []

        for site in self._history_source.load_site_catalog():
            usable = [
                observation
                for observation in self._history_source.load_observations(site.site_id)
                if observation.consumption_kw is not None
            ]
            if len(usable) < self._minimum_observations:
                sites_skipped.append(site.site_id)
                continue

            train, test = split_chronologically(usable, self._test_ratio)
            evaluations.append(self._evaluate_one_site(site.site_id, train, test))

        return EvaluationReport(evaluations=evaluations, sites_skipped=sites_skipped)

    def _evaluate_one_site(
        self, site_id: str, train: Sequence[Observation], test: Sequence[Observation]
    ) -> SiteEvaluation:
        profile = fit_hourly_profile(train)
        fallback_prediction = sum(cast(float, o.consumption_kw) for o in train) / len(train)
        baseline_predictions = [
            self._baseline_prediction(profile, observation, fallback_prediction)
            for observation in test
        ]

        forecaster = self._create_forecaster(site_id)
        forecaster.fit(train)
        model_predictions = forecaster.predict(test)

        truths = [cast(float, observation.consumption_kw) for observation in test]
        return evaluate_site(
            site_id,
            truths,
            model_predictions,
            baseline_predictions,
            forecaster.model_version,
            forecaster.estimator_name,
        )

    @staticmethod
    def _baseline_prediction(
        profile: HourlyProfile, observation: Observation, fallback_prediction: float
    ) -> float:
        # Une heure jamais vue a l'entrainement retombe sur la moyenne globale : sinon
        # l'evaluation perdrait ce point au lieu de sanctionner une baseline incomplete.
        predicted = predict_from_profile(profile, observation.timestamp)
        return predicted if predicted is not None else fallback_prediction
