from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from psycopg.errors import ForeignKeyViolation, UniqueViolation

from enervision_ml.load.errors import PersistenceError, UnknownSiteReferenceError
from enervision_ml.load.prediction_repository import insert_if_new, insert_many
from enervision_ml.records import PredictionRow

PREDICTION_ID = UUID("2f1c8b3a-5d47-4e21-9a6f-0c3b7e8d1a52")
TARGET_TIMESTAMP = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)
GENERATED_AT = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def build_prediction(threshold_kw: float = 150.0) -> PredictionRow:
    return PredictionRow(
        prediction_id=PREDICTION_ID,
        site_id="SITE001",
        target_timestamp=TARGET_TIMESTAMP,
        predicted_consumption_kw=87.3,
        threshold_kw=threshold_kw,
        model_version="scikit-learn==1.9.0+abcdef",
        timestamp=GENERATED_AT,
    )


def test_a_prediction_is_written_with_its_deterministic_identifier(connection: Any) -> None:
    insert_if_new(connection, build_prediction())

    parameters = connection.opened_cursor.parameters[0]
    assert parameters[0] == PREDICTION_ID
    assert parameters[1] == "SITE001"
    assert parameters[2] == TARGET_TIMESTAMP


def test_a_replayed_prediction_never_overwrites_the_first_write(connection: Any) -> None:
    insert_if_new(connection, build_prediction())

    statement = connection.opened_cursor.statements[0]
    assert "ON CONFLICT (site_id, target_timestamp, model_version" in statement
    assert "DO NOTHING" in statement
    assert "DO UPDATE" not in statement


def test_a_prediction_for_an_unknown_site_names_that_site(failing_connection: Any) -> None:
    connection = failing_connection(ForeignKeyViolation("site absent"))

    with pytest.raises(UnknownSiteReferenceError) as failure:
        insert_if_new(connection, build_prediction())

    assert failure.value.site_id == "SITE001"


def test_another_driver_error_is_not_mistaken_for_a_missing_site(failing_connection: Any) -> None:
    connection = failing_connection(UniqueViolation("contrainte inattendue"))

    with pytest.raises(PersistenceError) as failure:
        insert_if_new(connection, build_prediction())

    assert not isinstance(failure.value, UnknownSiteReferenceError)


def test_insert_many_writes_every_prediction_in_order(connection: Any) -> None:
    rows = [
        build_prediction(),
        PredictionRow(
            prediction_id=UUID("00000000-0000-0000-0000-000000000002"),
            site_id="SITE001",
            target_timestamp=TARGET_TIMESTAMP.replace(hour=12),
            predicted_consumption_kw=90.0,
            threshold_kw=150.0,
            model_version="scikit-learn==1.9.0+abcdef",
            timestamp=GENERATED_AT,
        ),
    ]

    insert_many(connection, rows)

    assert len(connection.opened_cursor.statements) == 2
