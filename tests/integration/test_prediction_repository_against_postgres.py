"""Verification du depot prediction contre un vrai PostgreSQL avec TimescaleDB.

FakeConnection enregistre la requete sans l'executer : il ne peut rien dire de la
contrainte d'unicite qui porte l'idempotence du service, et c'est justement elle que
ce test exerce pour de bon.

Exclu de `uv run pytest` par defaut. Pour le lancer :

    docker run -d --name g4_test_db -e POSTGRES_USER=g4_app -e POSTGRES_PASSWORD=test \\
      -e POSTGRES_DB=g4_db -p 5433:5432 \\
      -v "$PWD/../enervision-devops/db/init:/docker-entrypoint-initdb.d:ro" \\
      timescale/timescaledb:latest-pg16

    ENERVISION_TEST_DATABASE_URL=postgres://g4_app:test@localhost:5433/g4_db \\
      uv run pytest tests/integration -m integration
"""

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest

from enervision_ml.load.errors import UnknownSiteReferenceError
from enervision_ml.load.prediction_repository import insert_if_new
from enervision_ml.records import PredictionRow

pytestmark = pytest.mark.integration

DATABASE_URL_VARIABLE = "ENERVISION_TEST_DATABASE_URL"
TARGET_TIMESTAMP = datetime(2024, 6, 15, 15, 0, tzinfo=UTC)
GENERATED_AT = datetime(2024, 6, 15, 14, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def database_url() -> str:
    configured_url = os.environ.get(DATABASE_URL_VARIABLE)
    if configured_url is None:
        pytest.skip(f"{DATABASE_URL_VARIABLE} absente, voir le docstring du module")
    return configured_url


@pytest.fixture
def connection(database_url: str) -> Iterator[psycopg.Connection]:
    """Ouvre une transaction annulee a la fin : aucun test ne voit les donnees d'un autre."""
    opened = psycopg.connect(database_url, autocommit=False)
    try:
        yield opened
    finally:
        opened.rollback()
        opened.close()


@pytest.fixture
def known_site(connection: psycopg.Connection) -> str:
    site_id = "SITE_ML_TEST"
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO site (site_id, site_type, site_name, capacity_kw, status)
            VALUES (%s, 'office', 'Site de test integration ml', 500, 'active')
            ON CONFLICT (site_id) DO NOTHING
            """,
            (site_id,),
        )
    connection.commit()
    return site_id


def build_prediction(site_id: str) -> PredictionRow:
    return PredictionRow(
        prediction_id=uuid4(),
        site_id=site_id,
        target_timestamp=TARGET_TIMESTAMP,
        predicted_consumption_kw=87.3,
        threshold_kw=170.0,
        model_version="scikit-learn==1.9.0+test",
        timestamp=GENERATED_AT,
    )


def count_predictions(connection: psycopg.Connection, site_id: str) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM prediction WHERE site_id = %s", (site_id,))
        row = cursor.fetchone()
    return int(row[0]) if row is not None else 0


def test_the_schema_carries_the_constraint_the_service_relies_on(
    connection: psycopg.Connection,
) -> None:
    # Sans elle, ON CONFLICT est rejete par Postgres avec l'erreur 42P10.
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT conname FROM pg_constraint WHERE conname = 'uq_prediction_site_target_model'"
        )
        found = cursor.fetchone()

    assert found is not None, "contrainte uq_prediction_site_target_model absente : voir l'etape 5"


def test_a_replayed_prediction_is_absorbed_not_duplicated(
    connection: psycopg.Connection, known_site: str
) -> None:
    prediction = build_prediction(known_site)

    insert_if_new(connection, prediction)
    insert_if_new(connection, prediction)  # meme triplet (site, heure visee, version)

    assert count_predictions(connection, known_site) == 1


def test_a_prediction_for_an_unknown_site_is_rejected(connection: psycopg.Connection) -> None:
    with pytest.raises(UnknownSiteReferenceError):
        insert_if_new(connection, build_prediction("SITE_DOES_NOT_EXIST"))
