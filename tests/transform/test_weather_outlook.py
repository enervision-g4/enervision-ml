from datetime import UTC, datetime

from enervision_ml.records import Observation
from enervision_ml.transform.weather_outlook import build_climatology, project_weather


def make_observation(
    month: int, hour: int, day: int, temperature_celsius: float, humidity_percent: float
) -> Observation:
    return Observation(
        site_id="SITE001",
        timestamp=datetime(2024, month, day, hour, 0, tzinfo=UTC),
        consumption_kw=50.0,
        temperature_celsius=temperature_celsius,
        humidity_percent=humidity_percent,
    )


def test_the_climatology_averages_by_month_and_hour_not_calendar_day() -> None:
    observations = [
        make_observation(month=1, hour=14, day=1, temperature_celsius=10.0, humidity_percent=60.0),
        make_observation(month=1, hour=14, day=15, temperature_celsius=20.0, humidity_percent=70.0),
    ]

    climatology = build_climatology(observations)
    projection = project_weather(climatology, datetime(2024, 1, 28, 14, 0, tzinfo=UTC))

    assert projection.temperature_celsius == 15.0
    assert projection.humidity_percent == 65.0


def test_an_unobserved_month_hour_pair_has_no_projection() -> None:
    observations = [
        make_observation(month=1, hour=14, day=1, temperature_celsius=10.0, humidity_percent=60.0),
    ]

    climatology = build_climatology(observations)
    projection = project_weather(climatology, datetime(2024, 7, 3, 0, tzinfo=UTC))

    assert projection.temperature_celsius is None
    assert projection.humidity_percent is None


def test_the_climatology_ignores_missing_readings_independently_per_field() -> None:
    observations = [
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
            consumption_kw=50.0,
            temperature_celsius=None,
            humidity_percent=60.0,
        ),
    ]

    climatology = build_climatology(observations)
    projection = project_weather(climatology, datetime(2024, 1, 15, 14, 0, tzinfo=UTC))

    assert projection.temperature_celsius is None
    assert projection.humidity_percent == 60.0
