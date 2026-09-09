from datetime import UTC, datetime, timedelta

from enervision_ml.records import Observation
from enervision_ml.transform.aggregation import (
    aggregate_to_hourly,
    drop_partial_edges,
    infer_sampling_step,
)


def make_sample(
    minute_offset: int, hour: int = 8, consumption_kw: float = 10.0
) -> Observation:
    return Observation(
        site_id="SITE001",
        timestamp=datetime(2024, 1, 15, hour, 0, tzinfo=UTC) + timedelta(minutes=minute_offset),
        consumption_kw=consumption_kw,
        temperature_celsius=18.0,
        humidity_percent=60.0,
    )


def test_the_sampling_step_is_the_median_gap_not_the_first_one() -> None:
    # Un premier ecart de 15 min suivi d'ecarts reguliers de 60 min : se fier au
    # premier ecart donnerait un pas errone.
    timestamps = [
        datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
        datetime(2024, 1, 15, 8, 15, tzinfo=UTC),
        datetime(2024, 1, 15, 9, 15, tzinfo=UTC),
        datetime(2024, 1, 15, 10, 15, tzinfo=UTC),
        datetime(2024, 1, 15, 11, 15, tzinfo=UTC),
    ]

    step = infer_sampling_step(timestamps)

    assert step == timedelta(minutes=60)


def test_sixty_minute_measures_collapse_into_one_hourly_average() -> None:
    quarter_hourly_samples = [
        make_sample(0, consumption_kw=8.0),
        make_sample(15, consumption_kw=10.0),
        make_sample(30, consumption_kw=12.0),
        make_sample(45, consumption_kw=10.0),
    ]

    hourly = aggregate_to_hourly(quarter_hourly_samples)

    assert len(hourly) == 1
    assert hourly[0].timestamp == datetime(2024, 1, 15, 8, 0, tzinfo=UTC)
    assert hourly[0].consumption_kw == 10.0


def test_an_hourly_bucket_averages_only_the_known_values() -> None:
    samples = [
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
            consumption_kw=8.0,
            temperature_celsius=None,
            humidity_percent=60.0,
        ),
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 15, 8, 30, tzinfo=UTC),
            consumption_kw=12.0,
            temperature_celsius=20.0,
            humidity_percent=None,
        ),
    ]

    hourly = aggregate_to_hourly(samples)

    assert hourly[0].consumption_kw == 10.0
    assert hourly[0].temperature_celsius == 20.0
    assert hourly[0].humidity_percent == 60.0


def test_a_bucket_without_a_single_known_value_stays_none_not_zero() -> None:
    samples = [
        Observation(
            site_id="SITE001",
            timestamp=datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
            consumption_kw=None,
            temperature_celsius=None,
            humidity_percent=None,
        ),
    ]

    hourly = aggregate_to_hourly(samples)

    assert hourly[0].consumption_kw is None


def test_hourly_buckets_are_sorted_by_site_then_by_time() -> None:
    samples = [
        make_sample(0, hour=10),
        Observation(
            site_id="SITE002",
            timestamp=datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
            consumption_kw=5.0,
            temperature_celsius=15.0,
            humidity_percent=50.0,
        ),
        make_sample(0, hour=9),
    ]

    hourly = aggregate_to_hourly(samples)

    assert [(o.site_id, o.timestamp.hour) for o in hourly] == [
        ("SITE001", 9),
        ("SITE001", 10),
        ("SITE002", 8),
    ]


def test_drop_partial_edges_removes_only_the_boundary_bucket_missing_samples() -> None:
    # Fenetre de lecture qui commence a 8h15 : le godet de 8h n'a qu'un quart des
    # mesures attendues a un pas de 15 min. Celui de 9h en a bien quatre.
    samples = [
        make_sample(15, hour=8),
        make_sample(0, hour=9),
        make_sample(15, hour=9),
        make_sample(30, hour=9),
        make_sample(45, hour=9),
    ]

    kept = drop_partial_edges(samples, sampling_step=timedelta(minutes=15))

    assert {sample.timestamp.hour for sample in kept} == {9}


def test_drop_partial_edges_keeps_an_interior_bucket_thinned_only_by_nulls() -> None:
    # Le godet interieur (9h) n'a que deux valeurs connues sur quatre attendues, mais
    # ce sont des nulls, pas une fenetre de lecture tronquee : il n'est pas un bord.
    samples = [
        make_sample(0, hour=8),
        make_sample(15, hour=8),
        make_sample(30, hour=8),
        make_sample(45, hour=8),
        make_sample(0, hour=9),
        make_sample(30, hour=9),
        make_sample(0, hour=10),
        make_sample(15, hour=10),
        make_sample(30, hour=10),
        make_sample(45, hour=10),
    ]

    kept = drop_partial_edges(samples, sampling_step=timedelta(minutes=15))

    assert {sample.timestamp.hour for sample in kept} == {8, 9, 10}
