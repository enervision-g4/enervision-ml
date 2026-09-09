"""Journalisation des metriques d'evaluation vers MLflow Tracking.

Best-effort : une panne du serveur de suivi ne doit jamais faire echouer evaluate,
qui reste utilisable hors ligne (CSV, aucune infrastructure). Seules des metriques
sont journalisees, jamais de modele : ce service n'en persiste aucun (voir
model/forecaster.py), il n'y a donc rien a versionner cote artefacts.
"""

from datetime import datetime
from typing import Optional, Protocol, cast

from ..transform.evaluation import SiteEvaluation
from ..transform.forecast_accuracy import ForecastAccuracy
from .evaluation_run import EvaluationReport


class MlflowClientLike(Protocol):
    """Sous-ensemble du client MLflow reellement utilise ici."""

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> object:
        """Ouvre un run, imbrique dans le run courant si nested est vrai."""
        ...

    def log_param(self, key: str, value: object) -> None:
        """Enregistre un parametre sur le run courant."""
        ...

    def log_metric(self, key: str, value: float) -> None:
        """Enregistre une metrique sur le run courant."""
        ...

    def end_run(self) -> None:
        """Ferme le run courant."""
        ...


class WarningLoggerLike(Protocol):
    """Le seul appel de journalisation dont ce module a besoin.

    Injecte plutot que le logger structlog global du reste du paquet : aucun test
    de ce depot ne capture la sortie de structlog (pas de precedent caplog), alors
    qu'ici verifier qu'un avertissement est bien emis en cas de panne fait partie
    du contrat teste.
    """

    def warning(self, event: str, **kw: object) -> None:
        """Emet un avertissement structure."""
        ...


def create_mlflow_client(tracking_uri: str) -> MlflowClientLike:
    """Cree un client MLflow pointant vers le serveur de suivi indique.

    Args:
        tracking_uri: URL HTTP du serveur MLflow.

    Returns:
        Le module mlflow, dont l'API fluide (start_run/log_param/log_metric/
        end_run sont des fonctions de module) satisfait structurellement
        MlflowClientLike, au meme titre que create_connection() adapte psycopg.
    """
    import os

    # L'image de production n'embarque pas l'executable git ; sans ce reglage,
    # mlflow-skinny tente quand meme de detecter un depot a chaque run et
    # deverse plusieurs paragraphes d'avertissement GitPython dans la sortie de
    # evaluate, sans rapport avec le suivi lui-meme.
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return cast(MlflowClientLike, mlflow)


class ExperimentTrackingLogger:
    """Journalise un EvaluationReport vers MLflow, sans jamais faire echouer l'appelant.

    Attributes:
        client: Client MLflow, ou None pour desactiver silencieusement le suivi
            (MLFLOW_TRACKING_URI absent ou vide).
    """

    def __init__(self, client: Optional[MlflowClientLike], logger: WarningLoggerLike) -> None:
        """Prepare le journal.

        Args:
            client: Client MLflow a utiliser, ou None pour ne rien journaliser.
            logger: Destination des avertissements en cas d'echec du suivi.
        """
        self.client = client
        self._logger = logger

    def log_evaluation_report(
        self, report: EvaluationReport, source: str, test_ratio: float
    ) -> None:
        """Journalise chaque evaluation du rapport, si un client est configure.

        Un run parent porte les parametres du lot (source, test_ratio) ; chaque
        site evalue est un run imbrique portant ses propres parametres et
        metriques. Toute erreur (serveur injoignable, etc.) est avertie puis
        avalee : evaluate ne doit jamais echouer a cause du suivi.

        Le run parent est nomme explicitement (evaluate_<source>) plutot que de
        laisser MLflow lui attribuer un nom aleatoire (adjectif-animal) : sans
        ca, il est indiscernable des runs par site dans la liste plate de l'UI.

        Args:
            report: Rapport produit par EvaluationRun.run().
            source: Origine de l'historique evalue (csv ou database).
            test_ratio: Fraction de l'historique reservee au test.
        """
        if self.client is None:
            return

        try:
            self._log_report(self.client, report, source, test_ratio)
        except Exception as failure:
            self._logger.warning("mlflow_logging_failed", error=str(failure))

    @staticmethod
    def _log_report(
        client: MlflowClientLike, report: EvaluationReport, source: str, test_ratio: float
    ) -> None:
        client.start_run(run_name=f"evaluate_{source}")
        try:
            client.log_param("source", source)
            client.log_param("test_ratio", str(test_ratio))
            for site_evaluation in report.evaluations:
                ExperimentTrackingLogger._log_site(client, site_evaluation)
        finally:
            client.end_run()

    @staticmethod
    def _log_site(client: MlflowClientLike, site_evaluation: SiteEvaluation) -> None:
        client.start_run(run_name=site_evaluation.site_id, nested=True)
        try:
            client.log_param("site_id", site_evaluation.site_id)
            client.log_param("model_version", site_evaluation.model_version)
            client.log_metric("model_mae", site_evaluation.model_mae)
            client.log_metric("baseline_mae", site_evaluation.baseline_mae)
            client.log_metric("model_mape", site_evaluation.model_mape.value)
            client.log_metric("baseline_mape", site_evaluation.baseline_mape.value)
            client.log_metric("improvement_percent", site_evaluation.improvement_percent)
        finally:
            client.end_run()

    def log_forecast_accuracy(
        self,
        site_id: str,
        model_version: str,
        target_timestamp: datetime,
        accuracy: ForecastAccuracy,
    ) -> None:
        """Journalise la justesse d'une prevision de production desormais resolue.

        Un run par appel, jamais imbrique : contrairement a evaluate, chaque site est
        juge independamment a son propre rythme (voir orchestration/forecast_run.py),
        il n'y a pas de lot commun a rattacher en parent. Nomme comme les runs enfants
        d'evaluate (run_name=site_id) mais distingue par le parametre stage, pour que
        les deux ne se confondent pas dans l'IHM MLflow.

        Args:
            site_id: Site concerne.
            model_version: Empreinte du modele ayant produit la prevision jugee.
            target_timestamp: Heure visee par la prevision jugee.
            accuracy: Ecart mesure entre cette prevision et la mesure reelle.
        """
        if self.client is None:
            return

        try:
            self._log_forecast_accuracy(
                self.client, site_id, model_version, target_timestamp, accuracy
            )
        except Exception as failure:
            self._logger.warning("mlflow_logging_failed", error=str(failure))

    @staticmethod
    def _log_forecast_accuracy(
        client: MlflowClientLike,
        site_id: str,
        model_version: str,
        target_timestamp: datetime,
        accuracy: ForecastAccuracy,
    ) -> None:
        client.start_run(run_name=site_id)
        try:
            client.log_param("stage", "forecast")
            client.log_param("site_id", site_id)
            client.log_param("model_version", model_version)
            client.log_param("target_timestamp", target_timestamp.isoformat())
            client.log_metric("forecast_mae", accuracy.mae)
            if accuracy.mape is not None:
                client.log_metric("forecast_mape", accuracy.mape.value)
        finally:
            client.end_run()


def build_experiment_logger() -> ExperimentTrackingLogger:
    """Construit le journal a partir de l'environnement.

    Lit MLFLOW_TRACKING_URI directement, sans passer par ForecastSettings : c'est
    ce qui garde `evaluate --source csv` utilisable sans aucune configuration,
    y compris sans DATABASE_URL.

    Returns:
        Le journal, desactive si MLFLOW_TRACKING_URI est absent ou vide.
    """
    import os

    from ..logging_setup import get_logger

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    client = create_mlflow_client(tracking_uri) if tracking_uri else None
    return ExperimentTrackingLogger(client=client, logger=get_logger("experiment_tracking"))
