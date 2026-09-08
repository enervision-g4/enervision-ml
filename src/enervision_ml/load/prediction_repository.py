"""Ecriture des previsions, jamais deux fois la meme.

L'identifiant de chaque prevision est genere cote client (voir
transform/prediction_drafts.py) plutot que laisse a la base : ON CONFLICT DO NOTHING
n'a rien a renvoyer via RETURNING lorsqu'il absorbe un doublon, et forecast_run.py a
besoin de cet identifiant pour la recommandation associee. Cet ecart assume a la
convention ETL (ou l'UUID est laisse a la base) est documente ici plutot qu'enfoui.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Optional, cast
from uuid import UUID

import psycopg

from ..postgres_connection import ConnectionLike
from ..records import PredictionRow
from .errors import PersistenceError, UnknownSiteReferenceError

TABLE = "prediction"

INSERT_PREDICTION = """
    INSERT INTO prediction
        (prediction_id, site_id, target_timestamp, predicted_consumption_kw,
         threshold_kw, model_version, "timestamp")
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (site_id, target_timestamp, model_version, "timestamp") DO NOTHING
"""
"""Insertion idempotente. DO NOTHING et jamais DO UPDATE : la premiere ecriture gagne."""

SELECT_LATEST_PREDICTION_FOR_TARGET = """
    SELECT prediction_id, site_id, target_timestamp, predicted_consumption_kw,
           threshold_kw, model_version, "timestamp"
      FROM prediction
     WHERE site_id = %s AND target_timestamp = %s
     ORDER BY "timestamp" DESC
     LIMIT 1
"""
"""La plus recemment ecrite pour cette heure visee : celle au plus petit delai
d'anticipation, typiquement issue du lot precedent (voir forecast_run.py)."""


def insert_if_new(connection: ConnectionLike, prediction: PredictionRow) -> None:
    """Ecrit une prevision, sans effet si elle est deja en base.

    Args:
        connection: Connexion ouverte vers la base.
        prediction: Prevision a ecrire.

    Raises:
        UnknownSiteReferenceError: Si le site n'est pas dans le referentiel.
        PersistenceError: Si le pilote refuse l'ecriture pour une autre raison.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                INSERT_PREDICTION,
                (
                    prediction.prediction_id,
                    prediction.site_id,
                    prediction.target_timestamp,
                    prediction.predicted_consumption_kw,
                    prediction.threshold_kw,
                    prediction.model_version,
                    prediction.timestamp,
                ),
            )
    except psycopg.errors.ForeignKeyViolation as unknown_site:
        raise UnknownSiteReferenceError(TABLE, prediction.site_id) from unknown_site
    except psycopg.Error as refused_write:
        raise PersistenceError(TABLE, str(refused_write)) from refused_write


def insert_many(connection: ConnectionLike, predictions: Sequence[PredictionRow]) -> None:
    """Ecrit plusieurs previsions, dans l'ordre, sans effet pour celles deja en base.

    Args:
        connection: Connexion ouverte vers la base.
        predictions: Previsions a ecrire.

    Raises:
        UnknownSiteReferenceError: Si un site n'est pas dans le referentiel.
        PersistenceError: Si le pilote refuse une ecriture pour une autre raison.
    """
    for prediction in predictions:
        insert_if_new(connection, prediction)


def fetch_latest_for_target(
    connection: ConnectionLike, site_id: str, target_timestamp: datetime
) -> Optional[PredictionRow]:
    """Lit la prevision la plus recemment ecrite pour un site et une heure visee.

    Sert a juger une prevision passee contre la mesure reelle desormais connue (voir
    transform/forecast_accuracy.py), pas a servir des previsions au dashboard : c'est
    le role de enervision-api.

    Args:
        connection: Connexion ouverte vers la base.
        site_id: Site concerne.
        target_timestamp: Heure visee.

    Returns:
        La prevision la plus recemment generee pour cette heure, ou None si aucune
        n'a ete ecrite (site nouveau, ou lot precedent en echec).

    Raises:
        PersistenceError: Si le pilote refuse la lecture.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_LATEST_PREDICTION_FOR_TARGET, (site_id, target_timestamp))
            row = cursor.fetchone()
    except psycopg.Error as refused_read:
        raise PersistenceError(TABLE, str(refused_read)) from refused_read

    if row is None:
        return None

    (
        prediction_id,
        row_site_id,
        row_target_timestamp,
        predicted_consumption_kw,
        threshold_kw,
        model_version,
        generated_at,
    ) = row
    return PredictionRow(
        prediction_id=cast(UUID, prediction_id),
        site_id=cast(str, row_site_id),
        target_timestamp=cast(datetime, row_target_timestamp),
        predicted_consumption_kw=cast(float, predicted_consumption_kw),
        threshold_kw=cast(Optional[float], threshold_kw),
        model_version=cast(str, model_version),
        timestamp=cast(datetime, generated_at),
    )
