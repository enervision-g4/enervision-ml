from typing import Any

import pytest
from pydantic import ValidationError

from enervision_ml.config import ForecastSettings

CONFIGURABLE_VARIABLES = (
    "DATABASE_URL",
    "TRAINING_SOURCE",
    "CSV_PATH",
    "CSV_SOURCE_TIMEZONE",
    "FORECAST_INTERVAL_SECONDS",
    "HORIZON_HOURS",
    "MIN_TRAINING_HOURS",
    "THRESHOLD_RATIO",
    "LOG_LEVEL",
    "LOG_AS_JSON",
)

VALID_DATABASE_URL = "postgres://g4_app:secret@g4_db:5432/g4_db"


@pytest.fixture
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for variable_name in CONFIGURABLE_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)
    return monkeypatch


def build_settings(**environment: Any) -> ForecastSettings:
    return ForecastSettings(_env_file=None, **environment)


def test_loads_the_database_url_from_the_environment(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(database_url=VALID_DATABASE_URL)

    assert settings.database_url == VALID_DATABASE_URL


def test_a_missing_database_url_stops_the_startup(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        build_settings()


def test_a_database_url_without_a_known_scheme_is_refused(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        build_settings(database_url="mysql://u:p@h:3306/d")


def test_both_postgres_url_schemes_are_accepted(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    for url in ("postgres://u:p@h:5432/d", "postgresql://u:p@h:5432/d"):
        assert build_settings(database_url=url).database_url == url


def test_a_sqlalchemy_style_driver_suffix_is_stripped(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    # D'autres services du parc utilisent SQLAlchemy et ecrivent parfois ce schema
    # dans le DATABASE_URL partage entre tous les services : psycopg ne le comprend pas.
    settings = build_settings(database_url="postgresql+psycopg://g4_app:secret@g4_db:5432/g4_db")

    assert settings.database_url == "postgresql://g4_app:secret@g4_db:5432/g4_db"


def test_an_unknown_training_source_is_refused(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        build_settings(database_url=VALID_DATABASE_URL, training_source="parquet")


def test_a_csv_source_without_a_path_is_refused(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValidationError):
        build_settings(database_url=VALID_DATABASE_URL, training_source="csv")


def test_a_csv_source_with_a_path_is_accepted(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(
        database_url=VALID_DATABASE_URL, training_source="csv", csv_path="history.csv"
    )

    assert settings.csv_path == "history.csv"


def test_windows_carriage_returns_are_stripped_from_every_value(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(
        database_url=f"{VALID_DATABASE_URL}\r\n", csv_source_timezone="UTC\r"
    )

    assert settings.database_url == VALID_DATABASE_URL
    assert settings.csv_source_timezone == "UTC"


def test_every_setting_but_the_database_url_has_a_default(
    isolated_environment: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(database_url=VALID_DATABASE_URL)

    assert settings.training_source == "database"
    assert settings.csv_path is None
    assert settings.csv_source_timezone == "UTC"
    assert settings.forecast_interval_seconds == 3600
    assert settings.horizon_hours == 24
    assert settings.min_training_hours == 168
    assert settings.threshold_ratio == 0.85
    assert settings.log_level == "INFO"
    assert settings.log_as_json is True
