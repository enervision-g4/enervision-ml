import pytest

from enervision_ml.model.errors import NotEnoughSamplesError, UnfittedModelError
from enervision_ml.model.forecaster import MINIMUM_TRAINING_SAMPLES, ConsumptionForecaster

from .conftest import StubEstimator, make_observations


def test_the_model_version_is_set_before_fitting() -> None:
    forecaster = ConsumptionForecaster("SITE001", estimator=StubEstimator())

    assert forecaster.model_version
    assert not forecaster.is_fitted


def test_predict_before_fit_raises_unfitted_model_error() -> None:
    forecaster = ConsumptionForecaster("SITE001", estimator=StubEstimator())

    with pytest.raises(UnfittedModelError) as failure:
        forecaster.predict(make_observations(1))

    assert failure.value.site_id == "SITE001"


def test_fit_with_too_few_usable_samples_raises_not_enough_samples_error() -> None:
    forecaster = ConsumptionForecaster("SITE001", estimator=StubEstimator())
    too_few = make_observations(MINIMUM_TRAINING_SAMPLES - 1)

    with pytest.raises(NotEnoughSamplesError) as failure:
        forecaster.fit(too_few)

    assert failure.value.sample_count == MINIMUM_TRAINING_SAMPLES - 1


def test_fit_ignores_observations_without_a_known_consumption() -> None:
    estimator = StubEstimator()
    forecaster = ConsumptionForecaster("SITE001", estimator=estimator)
    observations = make_observations(MINIMUM_TRAINING_SAMPLES, consumption_kw=10.0)
    observations += make_observations(5, consumption_kw=None)

    forecaster.fit(observations)

    trained_target = estimator.fit_calls[0][1]
    assert len(trained_target) == MINIMUM_TRAINING_SAMPLES


def test_fit_marks_the_model_as_fitted() -> None:
    forecaster = ConsumptionForecaster("SITE001", estimator=StubEstimator())

    forecaster.fit(make_observations(MINIMUM_TRAINING_SAMPLES))

    assert forecaster.is_fitted


def test_predict_returns_one_value_per_observation_in_order() -> None:
    forecaster = ConsumptionForecaster("SITE001", estimator=StubEstimator(fixed_prediction=7.0))
    forecaster.fit(make_observations(MINIMUM_TRAINING_SAMPLES))

    predictions = forecaster.predict(make_observations(3))

    assert predictions == [7.0, 7.0, 7.0]
