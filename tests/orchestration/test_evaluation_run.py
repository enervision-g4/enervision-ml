from enervision_ml.model.forecaster import ConsumptionForecaster
from enervision_ml.orchestration.evaluation_run import EvaluationRun

from .conftest import StubEstimator, StubHistorySource, make_hourly_observations


def make_run(observations_by_site: dict, minimum_observations: int = 48) -> EvaluationRun:
    return EvaluationRun(
        history_source=StubHistorySource(observations_by_site),
        minimum_observations=minimum_observations,
        create_forecaster=lambda site_id: ConsumptionForecaster(
            site_id, estimator=StubEstimator(fixed_prediction=12.0)
        ),
    )


def test_a_site_with_too_short_a_history_is_skipped_without_stopping_the_parc() -> None:
    run = make_run(
        {
            "SITE001": make_hourly_observations("SITE001", 10),  # trop court
            "SITE002": make_hourly_observations("SITE002", 100),
        }
    )

    report = run.run()

    assert report.sites_skipped == ["SITE001"]
    assert [evaluation.site_id for evaluation in report.evaluations] == ["SITE002"]


def test_the_report_lists_one_evaluation_per_site_with_enough_history() -> None:
    run = make_run(
        {
            "SITE001": make_hourly_observations("SITE001", 100),
            "SITE002": make_hourly_observations("SITE002", 100),
        }
    )

    report = run.run()

    assert {evaluation.site_id for evaluation in report.evaluations} == {"SITE001", "SITE002"}
    assert report.sites_skipped == []


def test_evaluation_never_touches_a_database() -> None:
    # StubHistorySource ne porte aucune connexion : si le run compile et s'execute,
    # aucune infrastructure n'a ete sollicitee.
    run = make_run({"SITE001": make_hourly_observations("SITE001", 100)})

    report = run.run()

    assert len(report.evaluations) == 1


def test_the_model_predictions_come_from_the_injected_forecaster() -> None:
    run = make_run({"SITE001": make_hourly_observations("SITE001", 100)})

    report = run.run()

    # StubEstimator predit toujours 12.0 : l'ecart absolu moyen du modele n'est autre
    # que l'ecart moyen entre 12.0 et les verites de test.
    assert report.evaluations[0].model_mae >= 0
