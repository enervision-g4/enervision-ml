import math
from datetime import UTC, datetime, timedelta, timezone

from enervision_ml.records import Observation
from enervision_ml.transform.features import FEATURE_NAMES, build_feature_row


def make_observation(
    timestamp: datetime,
    consumption_kw: float = 100.0,
    temperature_celsius: float = 18.0,
    humidity_percent: float = 60.0,
) -> Observation:
    return Observation(
        site_id="SITE001",
        timestamp=timestamp,
        consumption_kw=consumption_kw,
        temperature_celsius=temperature_celsius,
        humidity_percent=humidity_percent,
    )


def test_a_saturday_is_flagged_as_a_weekend() -> None:
    saturday_noon = datetime(2024, 1, 13, 12, 0, tzinfo=UTC)

    row = build_feature_row(make_observation(saturday_noon))

    assert row[FEATURE_NAMES.index("is_weekend")] == 1.0


def test_a_tuesday_is_not_flagged_as_a_weekend() -> None:
    tuesday_noon = datetime(2024, 1, 16, 12, 0, tzinfo=UTC)

    row = build_feature_row(make_observation(tuesday_noon))

    assert row[FEATURE_NAMES.index("is_weekend")] == 0.0


def test_the_calendar_features_are_derived_from_the_utc_instant_not_the_local_one() -> None:
    # 23h30 dans un fuseau UTC+5 est 18h30 en UTC, la veille du jour local.
    late_local_evening = datetime(2024, 1, 15, 23, 30, tzinfo=timezone(timedelta(hours=5)))

    row = build_feature_row(make_observation(late_local_evening))

    assert row[FEATURE_NAMES.index("hour_of_day")] == 18.0
    assert row[FEATURE_NAMES.index("day_of_week")] == 0.0  # lundi en UTC, pas mardi


def test_a_missing_temperature_is_carried_as_a_gap_not_as_zero() -> None:
    observation = make_observation(
        datetime(2024, 1, 15, 12, 0, tzinfo=UTC), temperature_celsius=None
    )

    row = build_feature_row(observation)

    assert math.isnan(row[FEATURE_NAMES.index("temperature_celsius")])


def test_a_missing_humidity_is_carried_as_a_gap_not_as_zero() -> None:
    observation = make_observation(
        datetime(2024, 1, 15, 12, 0, tzinfo=UTC), humidity_percent=None
    )

    row = build_feature_row(observation)

    assert math.isnan(row[FEATURE_NAMES.index("humidity_percent")])


def test_the_feature_row_follows_the_declared_column_order() -> None:
    timestamp = datetime(2024, 3, 4, 9, 0, tzinfo=UTC)  # lundi 4 mars 2024, 9h

    row = build_feature_row(
        make_observation(timestamp, temperature_celsius=21.5, humidity_percent=55.0)
    )

    assert FEATURE_NAMES == (
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "temperature_celsius",
        "humidity_percent",
    )
    assert row == (9.0, 0.0, 0.0, 3.0, 21.5, 55.0)
