import pytest

from enervision_ml.extract.errors import DatabaseQueryError
from enervision_ml.extract.site_catalog import load_sites

from .conftest import FakeConnection


def test_a_site_without_declared_capacity_is_returned_with_an_unknown_capacity() -> None:
    connection = FakeConnection(rows=[("SITE001", 200), ("SITE002", None)])

    sites = load_sites(connection)

    assert sites[0].capacity_kw == 200
    assert sites[1].capacity_kw is None


def test_a_driver_failure_is_translated_into_an_extraction_error() -> None:
    connection = FakeConnection(failure=RuntimeError("connection reset"))

    with pytest.raises(DatabaseQueryError):
        load_sites(connection)
