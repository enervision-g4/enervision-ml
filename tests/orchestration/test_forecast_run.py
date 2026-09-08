from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from enervision_ml.model.forecaster import ConsumptionForecaster
from enervision_ml.orchestration.experiment_tracking import ExperimentTrackingLogger
from enervision_ml.orchestration.forecast_run import ForecastRun

from .conftest import (
    FakeConnection,
    FakeMlflowClient,
    FakeWarningLogger,
    StubEstimator,
    StubHistorySource,
    make_hourly_observations,
)

GENERATED_AT = datetime(2024, 2, 1, 10, 0, tzinfo=UTC)


def make_run(
    observations_by_site: dict,
    connection: FakeConnection,
    create_forecaster=None,
    minimum_training_hours: int = 48,
    experiment_logger=None,
) -> ForecastRun:
    return ForecastRun(
        history_source=StubHistorySource(observations_by_site),
        connection=connection,
        horizon_hours=24,
        minimum_training_hours=minimum_training_hours,
        threshold_ratio=0.85,
        create_forecaster=create_forecaster
        or (lambda site_id: ConsumptionForecaster(site_id, estimator=StubEstimator())),
        now=lambda: GENERATED_AT,
        experiment_logger=experiment_logger,
    )


def test_a_site_with_too_short_a_history_is_skipped_without_stopping_the_parc() -> None:
    connection = FakeConnection()
    run = make_run(
        {
            "SITE001": make_hourly_observations("SITE001", 10),  # trop court
            "SITE002": make_hourly_observations("SITE002", 100),
        },
        connection,
    )

    report = run.run()

    assert report.sites_skipped == ["SITE001"]
    assert report.sites_forecast == ["SITE002"]
    assert report.sites_failed == []


def test_a_successful_site_commits_its_predictions() -> None:
    connection = FakeConnection()
    run = make_run({"SITE001": make_hourly_observations("SITE001", 100)}, connection)

    report = run.run()

    assert report.sites_forecast == ["SITE001"]
    assert connection.commits == 1
    assert connection.rollbacks == 0
    # StubEstimator predit toujours 15.0, au-dessus de l'historique observe (10-14) :
    # les 24 previsions de l'horizon depassent toutes leur seuil de repli, donc 24
    # ecritures de prevision plus 24 de recommandation.
    assert len(connection.opened_cursor.statements) == 48
    assert report.recommendations_written == 24


def test_a_failing_site_is_rolled_back_and_the_next_one_still_runs() -> None:
    connection = FakeConnection()

    def create_forecaster(site_id: str) -> ConsumptionForecaster:
        if site_id == "SITE001":
            # Un estimateur dont fit() echoue simule une panne (donnees corrompues,
            # bibliotheque indisponible) sans avoir besoin d'un vrai defaut.
            class FailingEstimator(StubEstimator):
                def fit(self, features_matrix, target) -> None:
                    raise RuntimeError("training failed")

            return ConsumptionForecaster(site_id, estimator=FailingEstimator())
        return ConsumptionForecaster(site_id, estimator=StubEstimator())

    run = make_run(
        {
            "SITE001": make_hourly_observations("SITE001", 100),
            "SITE002": make_hourly_observations("SITE002", 100),
        },
        connection,
        create_forecaster=create_forecaster,
    )

    report = run.run()

    assert report.sites_failed == ["SITE001"]
    assert report.sites_forecast == ["SITE002"]
    assert connection.rollbacks == 1
    assert connection.commits == 1


def test_a_recommendation_is_never_committed_without_its_prediction() -> None:
    # L'ecriture de la recommandation echoue apres que les previsions ont ete
    # envoyees mais avant le commit : le rollback doit annuler les deux ensemble,
    # jamais laisser la prevision seule en base sans sa recommandation.
    connection = FakeConnection()
    original_execute = connection.opened_cursor.execute

    def failing_on_recommendation(statement, parameters=None):
        if "INSERT INTO recommendation" in statement:
            raise RuntimeError("recommendation write failed")
        return original_execute(statement, parameters)

    connection.opened_cursor.execute = failing_on_recommendation  # type: ignore[method-assign]

    run = make_run({"SITE001": make_hourly_observations("SITE001", 100)}, connection)
    report = run.run()

    assert report.sites_failed == ["SITE001"]
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_every_prediction_targets_an_hour_at_or_after_the_generation_instant() -> None:
    connection = FakeConnection()
    captured_prediction_parameters = []
    original_execute = connection.opened_cursor.execute

    def capturing_execute(statement, parameters=None):
        if "INSERT INTO prediction" in statement:
            captured_prediction_parameters.append(parameters)
        return original_execute(statement, parameters)

    connection.opened_cursor.execute = capturing_execute  # type: ignore[method-assign]

    run = make_run({"SITE001": make_hourly_observations("SITE001", 100)}, connection)
    run.run()

    # Le "timestamp" de generation est le dernier parametre d'un INSERT INTO prediction,
    # voir load/prediction_repository.py.
    generation_instants = {parameters[-1] for parameters in captured_prediction_parameters}
    assert generation_instants == {GENERATED_AT}


def test_a_previous_prediction_is_graded_against_the_new_actual_measurement() -> None:
    connection = FakeConnection()
    observations = make_hourly_observations("SITE001", 100)
    latest = observations[-1]
    mlflow_client = FakeMlflowClient()
    experiment_logger = ExperimentTrackingLogger(client=mlflow_client, logger=FakeWarningLogger())

    # Simule une prevision ecrite au lot precedent pour l'heure la plus recente
    # desormais connue, avec un modele different de celui qui va etre reentraine
    # ce tour-ci : c'est bien ce modele passe qui doit etre juge.
    connection.opened_cursor.fetchone = lambda: (
        uuid4(),
        "SITE001",
        latest.timestamp,
        latest.consumption_kw + 2.0,
        None,
        "scikit-learn==1.8.0+oldmodel",
        latest.timestamp - timedelta(hours=1),
    )

    run = make_run({"SITE001": observations}, connection, experiment_logger=experiment_logger)
    run.run()

    assert ("SITE001", False) in mlflow_client.started_runs
    assert ("stage", "forecast") in mlflow_client.logged_params
    assert ("model_version", "scikit-learn==1.8.0+oldmodel") in mlflow_client.logged_params
    assert ("forecast_mae", pytest.approx(2.0)) in mlflow_client.logged_metrics


def test_no_run_is_logged_when_no_previous_prediction_exists() -> None:
    # FakeCursor.fetchone() rend None par defaut : aucune prevision precedente.
    connection = FakeConnection()
    mlflow_client = FakeMlflowClient()
    experiment_logger = ExperimentTrackingLogger(client=mlflow_client, logger=FakeWarningLogger())

    run = make_run(
        {"SITE001": make_hourly_observations("SITE001", 100)},
        connection,
        experiment_logger=experiment_logger,
    )
    run.run()

    assert mlflow_client.started_runs == []


def test_a_failing_accuracy_lookup_does_not_prevent_the_site_from_forecasting() -> None:
    connection = FakeConnection()

    def failing_fetchone():
        raise RuntimeError("database unavailable")

    connection.opened_cursor.fetchone = failing_fetchone
    experiment_logger = ExperimentTrackingLogger(
        client=FakeMlflowClient(), logger=FakeWarningLogger()
    )

    run = make_run(
        {"SITE001": make_hourly_observations("SITE001", 100)},
        connection,
        experiment_logger=experiment_logger,
    )
    report = run.run()

    assert report.sites_forecast == ["SITE001"]
    assert report.sites_failed == []
    # La lecture ratee annule sa propre transaction, pour que l'ecriture des
    # nouvelles previsions reparte d'une transaction saine.
    assert connection.rollbacks == 1
    assert connection.commits == 1
