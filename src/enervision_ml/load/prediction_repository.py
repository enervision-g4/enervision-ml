"""Ecriture des previsions, jamais deux fois la meme.

L'identifiant de chaque prevision est genere cote client (voir
transform/prediction_drafts.py) plutot que laisse a la base : ON CONFLICT DO NOTHING
n'a rien a renvoyer via RETURNING lorsqu'il absorbe un doublon, et forecast_run.py a
besoin de cet identifiant pour la recommandation associee. Cet ecart assume a la
convention ETL (ou l'UUID est laisse a la base) est documente ici plutot qu'enfoui.
"""

from collections.abc import Sequence

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
