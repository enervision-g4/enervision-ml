from datetime import UTC, datetime

from enervision_ml.transform.prediction_drafts import (
    build_prediction_id,
    build_prediction_rows,
    build_target_timestamps,
)

MODEL_VERSION = "scikit-learn==1.9.0+abcdef123456"


def test_the_horizon_starts_at_the_next_whole_hour() -> None:
    now = datetime(2024, 1, 15, 10, 37, tzinfo=UTC)

    timestamps = build_target_timestamps(now, horizon_hours=3)

    assert timestamps == [
        datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
        datetime(2024, 1, 15, 12, 0, tzinfo=UTC),
        datetime(2024, 1, 15, 13, 0, tzinfo=UTC),
    ]


def test_the_same_forecast_replayed_keeps_the_same_identifier() -> None:
    target_timestamp = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)

    first_run = build_prediction_id("SITE001", target_timestamp, MODEL_VERSION)
    replay = build_prediction_id("SITE001", target_timestamp, MODEL_VERSION)

    assert first_run == replay


def test_a_different_target_hour_yields_a_different_identifier() -> None:
    identifier_at_11h = build_prediction_id(
        "SITE001", datetime(2024, 1, 15, 11, 0, tzinfo=UTC), MODEL_VERSION
    )
    identifier_at_12h = build_prediction_id(
        "SITE001", datetime(2024, 1, 15, 12, 0, tzinfo=UTC), MODEL_VERSION
    )

    assert identifier_at_11h != identifier_at_12h


def test_every_prediction_of_one_run_carries_the_same_generation_instant() -> None:
    generated_at = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
    target_timestamps = build_target_timestamps(generated_at, horizon_hours=3)

    rows = build_prediction_rows(
        site_id="SITE001",
        target_timestamps=target_timestamps,
        predicted_consumption_kw=[10.0, 20.0, 30.0],
        threshold_kw=100.0,
        model_version=MODEL_VERSION,
        generated_at=generated_at,
    )

    assert all(row.timestamp == generated_at for row in rows)


def test_each_row_pairs_its_target_hour_with_its_own_prediction() -> None:
    generated_at = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
    target_timestamps = build_target_timestamps(generated_at, horizon_hours=2)

    rows = build_prediction_rows(
        site_id="SITE001",
        target_timestamps=target_timestamps,
        predicted_consumption_kw=[10.0, 20.0],
        threshold_kw=None,
        model_version=MODEL_VERSION,
        generated_at=generated_at,
    )

    assert rows[0].target_timestamp == target_timestamps[0]
    assert rows[0].predicted_consumption_kw == 10.0
    assert rows[1].predicted_consumption_kw == 20.0
    assert rows[0].threshold_kw is None
