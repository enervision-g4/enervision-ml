from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from psycopg.errors import ForeignKeyViolation, UniqueViolation

from enervision_ml.load.errors import PersistenceError, UnknownSiteReferenceError
from enervision_ml.load.recommendation_repository import insert_if_new, insert_many
from enervision_ml.records import RecommendationRow

RECOMMENDATION_ID = UUID("11111111-1111-1111-1111-111111111111")
PREDICTION_ID = UUID("2f1c8b3a-5d47-4e21-9a6f-0c3b7e8d1a52")
GENERATED_AT = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def build_recommendation() -> RecommendationRow:
    return RecommendationRow(
        recommendation_id=RECOMMENDATION_ID,
        site_id="SITE001",
        prediction_id=PREDICTION_ID,
        timestamp=GENERATED_AT,
        action_description="Consommation prevue au-dessus du seuil.",
        status="open",
    )


def test_a_recommendation_is_written_with_its_prediction_reference(connection: Any) -> None:
    insert_if_new(connection, build_recommendation())

    parameters = connection.opened_cursor.parameters[0]
    assert parameters[0] == RECOMMENDATION_ID
    assert parameters[1] == "SITE001"
    assert parameters[2] == PREDICTION_ID


def test_a_replayed_recommendation_never_overwrites_the_first_write(connection: Any) -> None:
    insert_if_new(connection, build_recommendation())

    statement = connection.opened_cursor.statements[0]
    assert 'ON CONFLICT (prediction_id, "timestamp")' in statement
    assert "DO NOTHING" in statement
    assert "DO UPDATE" not in statement


def test_a_recommendation_for_an_unknown_site_names_that_site(failing_connection: Any) -> None:
    connection = failing_connection(ForeignKeyViolation("site absent"))

    with pytest.raises(UnknownSiteReferenceError) as failure:
        insert_if_new(connection, build_recommendation())

    assert failure.value.site_id == "SITE001"


def test_another_driver_error_is_not_mistaken_for_a_missing_site(failing_connection: Any) -> None:
    connection = failing_connection(UniqueViolation("contrainte inattendue"))

    with pytest.raises(PersistenceError) as failure:
        insert_if_new(connection, build_recommendation())

    assert not isinstance(failure.value, UnknownSiteReferenceError)


def test_insert_many_writes_every_recommendation_in_order(connection: Any) -> None:
    rows = [
        build_recommendation(),
        RecommendationRow(
            recommendation_id=UUID("22222222-2222-2222-2222-222222222222"),
            site_id="SITE001",
            prediction_id=UUID("00000000-0000-0000-0000-000000000002"),
            timestamp=GENERATED_AT,
            action_description="Autre creneau au-dessus du seuil.",
            status="open",
        ),
    ]

    insert_many(connection, rows)

    assert len(connection.opened_cursor.statements) == 2
