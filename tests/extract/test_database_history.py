from datetime import UTC, datetime

import pytest

from enervision_ml.extract.database_history import (
    SELECT_HOURLY_OBSERVATIONS_FOR_SITE,
    DatabaseHistorySource,
)
from enervision_ml.extract.errors import DatabaseQueryError

from .conftest import FakeConnection


def test_the_hourly_bucket_is_an_average_of_kilowatts_never_a_sum_of_kilowatt_hours() -> None:
    # Une somme donnerait un creux fantome des qu'une heure n'a que quelques mesures ;
    # seule une moyenne reste une puissance correcte dans ce cas.
    assert "avg(consumption_kw)" in SELECT_HOURLY_OBSERVATIONS_FOR_SITE
    assert "sum(consumption_kw)" not in SELECT_HOURLY_OBSERVATIONS_FOR_SITE


def test_a_site_filter_uses_a_dedicated_statement_rather_than_a_null_parameter() -> None:
    # site_id = %s directement, jamais une clause (site_id = %s OR %s IS NULL) qui
    # empecherait Postgres d'utiliser l'index sur site_id.
    assert "site_id = %s" in SELECT_HOURLY_OBSERVATIONS_FOR_SITE
    assert "IS NULL" not in SELECT_HOURLY_OBSERVATIONS_FOR_SITE


def test_load_observations_sends_the_site_id_as_a_bound_parameter() -> None:
    connection = FakeConnection(rows=[])
    source = DatabaseHistorySource(connection)

    source.load_observations("SITE001")

    assert connection.opened_cursor.parameters[0][0] == "SITE001"


def test_load_observations_converts_rows_into_hourly_observations() -> None:
    connection = FakeConnection(
        rows=[(datetime(2024, 1, 15, 8, 0, tzinfo=UTC), 42.5, 18.0, 60.0, 4)]
    )
    source = DatabaseHistorySource(connection)

    observations = source.load_observations("SITE001")

    assert len(observations) == 1
    assert observations[0].site_id == "SITE001"
    assert observations[0].timestamp == datetime(2024, 1, 15, 8, 0, tzinfo=UTC)
    assert observations[0].consumption_kw == 42.5


def test_a_driver_failure_is_translated_into_an_extraction_error() -> None:
    connection = FakeConnection(failure=RuntimeError("connection reset"))
    source = DatabaseHistorySource(connection)

    with pytest.raises(DatabaseQueryError) as failure:
        source.load_observations("SITE001")

    assert failure.value.context == "SITE001"
