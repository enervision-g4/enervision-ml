from datetime import UTC, datetime

import pytest

from enervision_ml.records import Observation
from enervision_ml.transform.baseline import fit_hourly_profile, predict_from_profile


def make_observation(hour: int, day: int, consumption_kw: float) -> Observation:
    return Observation(
        site_id="SITE001",
        timestamp=datetime(2024, 1, day, hour, 0, tzinfo=UTC),
        consumption_kw=consumption_kw,
        temperature_celsius=18.0,
        humidity_percent=60.0,
    )


def test_fit_hourly_profile_averages_known_values_by_utc_hour() -> None:
    observations = [
        make_observation(hour=8, day=1, consumption_kw=10.0),
        make_observation(hour=8, day=2, consumption_kw=20.0),
        make_observation(hour=20, day=1, consumption_kw=100.0),
    ]

    profile = fit_hourly_profile(observations)

    assert profile.mean_by_hour[8] == 15.0
    assert profile.mean_by_hour[20] == 100.0


def test_fit_hourly_profile_ignores_observations_without_a_known_consumption() -> None:
    observations = [
        make_observation(hour=8, day=1, consumption_kw=10.0),
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 2, 8, 0, tzinfo=UTC),
            consumption_kw=None,
            temperature_celsius=18.0,
            humidity_percent=60.0,
        ),
    ]

    profile = fit_hourly_profile(observations)

    assert profile.mean_by_hour[8] == 10.0


def test_fitting_without_any_known_consumption_is_refused() -> None:
    observations = [
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 1, 8, 0, tzinfo=UTC),
            consumption_kw=None,
            temperature_celsius=18.0,
            humidity_percent=60.0,
        ),
    ]

    with pytest.raises(ValueError, match="known consumption"):
        fit_hourly_profile(observations)


def test_predict_from_profile_returns_the_hourly_mean() -> None:
    profile = fit_hourly_profile([make_observation(hour=8, day=1, consumption_kw=10.0)])

    prediction = predict_from_profile(profile, datetime(2024, 3, 5, 8, 0, tzinfo=UTC))

    assert prediction == 10.0


def test_predict_from_profile_returns_none_for_an_unobserved_hour() -> None:
    profile = fit_hourly_profile([make_observation(hour=8, day=1, consumption_kw=10.0)])

    prediction = predict_from_profile(profile, datetime(2024, 3, 5, 3, 0, tzinfo=UTC))

    assert prediction is None
