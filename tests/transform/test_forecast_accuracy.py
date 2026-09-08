from datetime import UTC, datetime, timedelta

import pytest

from enervision_ml.records import Observation
from enervision_ml.transform.forecast_accuracy import (
    compute_forecast_accuracy,
    most_recently_resolved_observation,
)


def make_observation(hours_offset: int, consumption_kw: float = 10.0) -> Observation:
    return Observation(
        site_id="SITE001",
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hours_offset),
        consumption_kw=consumption_kw,
        temperature_celsius=18.0,
        humidity_percent=60.0,
    )


def test_the_most_recent_observation_is_found_regardless_of_input_order() -> None:
    observations = [make_observation(2), make_observation(0), make_observation(1)]

    latest = most_recently_resolved_observation(observations)

    assert latest is not None
    assert latest.timestamp == make_observation(2).timestamp


def test_an_empty_history_has_no_resolved_observation() -> None:
    assert most_recently_resolved_observation([]) is None


def test_forecast_accuracy_reports_the_absolute_and_relative_gap() -> None:
    accuracy = compute_forecast_accuracy(predicted_consumption_kw=12.0, actual_consumption_kw=10.0)

    assert accuracy.mae == pytest.approx(2.0)
    assert accuracy.mape is not None
    assert accuracy.mape.value == pytest.approx(20.0)


def test_a_near_zero_actual_value_leaves_the_percentage_error_absent() -> None:
    # Une mesure reelle quasi nulle rendrait l'ecart relatif arbitrairement grand
    # pour un ecart absolu negligeable (voir NEAR_ZERO_TRUTH_KW).
    accuracy = compute_forecast_accuracy(predicted_consumption_kw=1.0, actual_consumption_kw=0.05)

    assert accuracy.mae == pytest.approx(0.95)
    assert accuracy.mape is None
