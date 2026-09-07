from datetime import UTC, datetime
from pathlib import Path

import pytest

from enervision_ml.extract.csv_history import CsvHistorySource
from enervision_ml.extract.errors import MissingColumnError

from .conftest import write_csv

FULL_HEADER = [
    "timestamp",
    "site_id",
    "site_type",
    "site_name",
    "consumption_kwh",
    "consumption_euros",
    "temperature_celsius",
    "humidity_percent",
    "solar_irradiance_wm2",
    "hour",
    "day_of_week",
    "day_name",
    "month",
    "is_weekend",
    "is_working_hours",
]


def make_row(
    timestamp: str,
    site_id: str = "SITE001",
    consumption_kwh: str = "10.0",
    temperature_celsius: str = "18.0",
    humidity_percent: str = "60",
    precomputed_hour: str = "",
) -> list[str]:
    hour_column = precomputed_hour or timestamp[11:13].lstrip("0") or "0"
    return [
        timestamp, site_id, "office", "Bureau", consumption_kwh, "1.5",
        temperature_celsius, humidity_percent, "0", hour_column, "0", "Monday", "1", "0", "0",
    ]


def test_the_precomputed_calendar_columns_are_ignored_in_favour_of_the_utc_timestamp(
    tmp_path: Path,
) -> None:
    # La colonne "hour" precalculee est volontairement fausse (23 au lieu de 3) :
    # la lecture doit s'appuyer sur le seul timestamp.
    csv_path = write_csv(
        tmp_path / "wrong_precomputed_hour.csv",
        FULL_HEADER,
        [make_row("2024-01-15 03:00:00", precomputed_hour="23")],
    )

    source = CsvHistorySource(str(csv_path))
    observations = source.load_observations("SITE001")

    assert observations[0].timestamp == datetime(2024, 1, 15, 3, 0, tzinfo=UTC)


def test_a_missing_mandatory_column_names_it(tmp_path: Path) -> None:
    incomplete_header = [name for name in FULL_HEADER if name != "consumption_kwh"]
    csv_path = write_csv(
        tmp_path / "incomplete.csv",
        incomplete_header,
        [[value for name, value in zip(FULL_HEADER, make_row("2024-01-15 00:00:00"), strict=True)
          if name != "consumption_kwh"]],
    )

    source = CsvHistorySource(str(csv_path))

    with pytest.raises(MissingColumnError) as failure:
        source.load_observations("SITE001")

    assert failure.value.column_name == "consumption_kwh"


def test_a_naive_timestamp_is_anchored_to_the_declared_timezone(tmp_path: Path) -> None:
    # Minuit a Paris en janvier (UTC+1, hors periode d'ete) vaut 23h la veille en UTC.
    csv_path = write_csv(
        tmp_path / "local_time.csv",
        FULL_HEADER,
        [
            make_row("2024-01-15 00:00:00"),
            make_row("2024-01-15 01:00:00"),
        ],
    )

    source = CsvHistorySource(str(csv_path), source_timezone="Europe/Paris")
    observations = source.load_observations("SITE001")

    assert observations[0].timestamp == datetime(2024, 1, 14, 23, 0, tzinfo=UTC)


def test_kilowatt_hours_are_converted_to_kilowatts_using_the_sampling_step(
    tmp_path: Path,
) -> None:
    # Pas source de 30 min : 5 kWh sur une demi-heure valent 10 kW de puissance
    # moyenne, pas 5 kW comme le supposerait une lecture qui ignore le pas.
    csv_path = write_csv(
        tmp_path / "half_hourly.csv",
        FULL_HEADER,
        [
            make_row("2024-01-15 00:00:00", consumption_kwh="5.0"),
            make_row("2024-01-15 00:30:00", consumption_kwh="5.0", precomputed_hour="0"),
        ],
    )

    source = CsvHistorySource(str(csv_path))
    observations = source.load_observations("SITE001")

    assert len(observations) == 1
    assert observations[0].consumption_kw == 10.0


def test_the_site_catalog_lists_every_distinct_site_without_a_known_capacity(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(
        tmp_path / "two_sites.csv",
        FULL_HEADER,
        [
            make_row("2024-01-15 00:00:00", site_id="SITE001"),
            make_row("2024-01-15 01:00:00", site_id="SITE001"),
            make_row("2024-01-15 00:00:00", site_id="SITE002"),
        ],
    )

    source = CsvHistorySource(str(csv_path))
    catalog = source.load_site_catalog()

    assert {site.site_id for site in catalog} == {"SITE001", "SITE002"}
    assert all(site.capacity_kw is None for site in catalog)


def test_an_unknown_site_returns_no_observation(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path / "one_site.csv",
        FULL_HEADER,
        [make_row("2024-01-15 00:00:00", site_id="SITE001")],
    )

    source = CsvHistorySource(str(csv_path))

    assert source.load_observations("SITE999") == []
