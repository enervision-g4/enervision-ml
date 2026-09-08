"""Ecriture des recommandations, jamais deux fois la meme.

Contrairement a prediction_repository.py, recommendation_id est laisse a la base : rien
ne depend de cet identifiant en aval, la convention ETL par defaut s'applique donc ici
sans ecart. L'idempotence repose sur (prediction_id, "timestamp"), voir
enervision-devops/db/migrations/002_add_recommendation_unique_constraint.sql.
"""

from collections.abc import Sequence

import psycopg

from ..postgres_connection import ConnectionLike
from ..records import RecommendationRow
from .errors import PersistenceError, UnknownSiteReferenceError

TABLE = "recommendation"

INSERT_RECOMMENDATION = """
    INSERT INTO recommendation
        (recommendation_id, site_id, prediction_id, "timestamp", action_description, status)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON CONFLICT (prediction_id, "timestamp") DO NOTHING
"""
"""Insertion idempotente. DO NOTHING et jamais DO UPDATE : la premiere ecriture gagne."""


def insert_if_new(connection: ConnectionLike, recommendation: RecommendationRow) -> None:
    """Ecrit une recommandation, sans effet si elle est deja en base.

    Args:
        connection: Connexion ouverte vers la base.
        recommendation: Recommandation a ecrire.

    Raises:
        UnknownSiteReferenceError: Si le site n'est pas dans le referentiel.
        PersistenceError: Si le pilote refuse l'ecriture pour une autre raison.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                INSERT_RECOMMENDATION,
                (
                    recommendation.recommendation_id,
                    recommendation.site_id,
                    recommendation.prediction_id,
                    recommendation.timestamp,
                    recommendation.action_description,
                    recommendation.status,
                ),
            )
    except psycopg.errors.ForeignKeyViolation as unknown_site:
        raise UnknownSiteReferenceError(TABLE, recommendation.site_id) from unknown_site
    except psycopg.Error as refused_write:
        raise PersistenceError(TABLE, str(refused_write)) from refused_write


def insert_many(connection: ConnectionLike, recommendations: Sequence[RecommendationRow]) -> None:
    """Ecrit plusieurs recommandations, dans l'ordre, sans effet pour celles deja en base.

    Args:
        connection: Connexion ouverte vers la base.
        recommendations: Recommandations a ecrire.

    Raises:
        UnknownSiteReferenceError: Si un site n'est pas dans le referentiel.
        PersistenceError: Si le pilote refuse une ecriture pour une autre raison.
    """
    for recommendation in recommendations:
        insert_if_new(connection, recommendation)
