from datetime import UTC, datetime

from enervision_ml.orchestration.evaluation_run import EvaluationReport
from enervision_ml.orchestration.experiment_tracking import ExperimentTrackingLogger
from enervision_ml.transform.evaluation import MapeResult, SiteEvaluation
from enervision_ml.transform.forecast_accuracy import ForecastAccuracy

from .conftest import FakeMlflowClient, FakeWarningLogger


def make_evaluation(site_id: str) -> SiteEvaluation:
    return SiteEvaluation(
        site_id=site_id,
        model_version="scikit-learn==1.9.0+abc123",
        model_mae=5.5,
        baseline_mae=10.2,
        model_mape=MapeResult(value=3.1, excluded_count=0),
        baseline_mape=MapeResult(value=8.4, excluded_count=1),
        improvement_percent=46.1,
    )


def test_logging_is_a_noop_when_no_tracking_uri_is_configured() -> None:
    warning_logger = FakeWarningLogger()
    tracker = ExperimentTrackingLogger(client=None, logger=warning_logger)
    report = EvaluationReport(evaluations=[make_evaluation("SITE001")], sites_skipped=[])

    tracker.log_evaluation_report(report, source="csv", test_ratio=0.2)

    assert warning_logger.warnings == []


def test_logging_never_raises_when_the_client_fails() -> None:
    client = FakeMlflowClient(raise_on="start_run")
    warning_logger = FakeWarningLogger()
    tracker = ExperimentTrackingLogger(client=client, logger=warning_logger)
    report = EvaluationReport(evaluations=[make_evaluation("SITE001")], sites_skipped=[])

    tracker.log_evaluation_report(report, source="csv", test_ratio=0.2)

    assert len(warning_logger.warnings) == 1
    assert warning_logger.warnings[0][0] == "mlflow_logging_failed"


def test_a_successful_run_logs_one_parent_and_one_child_run_per_site() -> None:
    client = FakeMlflowClient()
    tracker = ExperimentTrackingLogger(client=client, logger=FakeWarningLogger())
    report = EvaluationReport(
        evaluations=[make_evaluation("SITE001"), make_evaluation("SITE002")],
        sites_skipped=[],
    )

    tracker.log_evaluation_report(report, source="csv", test_ratio=0.2)

    # Un run parent nomme explicitement (nested=False), un run enfant par site
    # (nested=True) nomme par son site_id.
    assert client.started_runs[0] == ("evaluate_csv", False)
    assert client.started_runs[1:] == [("SITE001", True), ("SITE002", True)]

    assert ("source", "csv") in client.logged_params
    assert ("test_ratio", "0.2") in client.logged_params
    assert ("site_id", "SITE001") in client.logged_params
    assert ("model_version", "scikit-learn==1.9.0+abc123") in client.logged_params

    assert ("model_mae", 5.5) in client.logged_metrics
    assert ("baseline_mae", 10.2) in client.logged_metrics
    assert ("model_mape", 3.1) in client.logged_metrics
    assert ("baseline_mape", 8.4) in client.logged_metrics
    assert ("improvement_percent", 46.1) in client.logged_metrics

    # Un parent + deux enfants = trois runs ouverts, trois runs fermes.
    assert client.ended_run_count == 3


def test_all_runs_are_ended_even_if_a_site_fails_to_log() -> None:
    client = FakeMlflowClient(raise_on="log_metric")
    warning_logger = FakeWarningLogger()
    tracker = ExperimentTrackingLogger(client=client, logger=warning_logger)
    report = EvaluationReport(evaluations=[make_evaluation("SITE001")], sites_skipped=[])

    tracker.log_evaluation_report(report, source="csv", test_ratio=0.2)

    assert len(warning_logger.warnings) == 1
    # Le run enfant a ete ouvert avant l'echec, il ne doit pas rester ouvert.
    assert client.ended_run_count >= 1


def test_forecast_accuracy_logging_is_a_noop_when_no_tracking_uri_is_configured() -> None:
    warning_logger = FakeWarningLogger()
    tracker = ExperimentTrackingLogger(client=None, logger=warning_logger)

    tracker.log_forecast_accuracy(
        site_id="SITE001",
        model_version="scikit-learn==1.9.0+abc123",
        target_timestamp=datetime(2024, 2, 1, 10, 0, tzinfo=UTC),
        accuracy=ForecastAccuracy(mae=2.0, mape=MapeResult(value=20.0, excluded_count=0)),
    )

    assert warning_logger.warnings == []


def test_forecast_accuracy_is_logged_as_a_standalone_run_tagged_forecast() -> None:
    client = FakeMlflowClient()
    tracker = ExperimentTrackingLogger(client=client, logger=FakeWarningLogger())

    tracker.log_forecast_accuracy(
        site_id="SITE001",
        model_version="scikit-learn==1.9.0+abc123",
        target_timestamp=datetime(2024, 2, 1, 10, 0, tzinfo=UTC),
        accuracy=ForecastAccuracy(mae=2.0, mape=MapeResult(value=20.0, excluded_count=0)),
    )

    # Un seul run, jamais imbrique : pas de lot commun a rattacher en parent.
    assert client.started_runs == [("SITE001", False)]
    assert ("stage", "forecast") in client.logged_params
    assert ("site_id", "SITE001") in client.logged_params
    assert ("model_version", "scikit-learn==1.9.0+abc123") in client.logged_params
    assert ("forecast_mae", 2.0) in client.logged_metrics
    assert ("forecast_mape", 20.0) in client.logged_metrics
    assert client.ended_run_count == 1


def test_forecast_accuracy_skips_the_percentage_metric_when_absent() -> None:
    client = FakeMlflowClient()
    tracker = ExperimentTrackingLogger(client=client, logger=FakeWarningLogger())

    tracker.log_forecast_accuracy(
        site_id="SITE001",
        model_version="scikit-learn==1.9.0+abc123",
        target_timestamp=datetime(2024, 2, 1, 10, 0, tzinfo=UTC),
        accuracy=ForecastAccuracy(mae=0.95, mape=None),
    )

    assert ("forecast_mae", 0.95) in client.logged_metrics
    assert not any(key == "forecast_mape" for key, _ in client.logged_metrics)


def test_forecast_accuracy_logging_never_raises_when_the_client_fails() -> None:
    client = FakeMlflowClient(raise_on="start_run")
    warning_logger = FakeWarningLogger()
    tracker = ExperimentTrackingLogger(client=client, logger=warning_logger)

    tracker.log_forecast_accuracy(
        site_id="SITE001",
        model_version="scikit-learn==1.9.0+abc123",
        target_timestamp=datetime(2024, 2, 1, 10, 0, tzinfo=UTC),
        accuracy=ForecastAccuracy(mae=2.0, mape=None),
    )

    assert len(warning_logger.warnings) == 1
    assert warning_logger.warnings[0][0] == "mlflow_logging_failed"
