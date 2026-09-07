"""Referentiel des sites, source de capacity_kw pour le calcul du seuil d'alerte."""

from collections.abc import Sequence
from typing import Optional

from ..postgres_connection import ConnectionLike
from ..records import SiteReference
from .errors import DatabaseQueryError

SELECT_SITE_CATALOG = """
    SELECT site_id, capacity_kw
      FROM site
     ORDER BY site_id
"""
"""Un site par ligne : capacity_kw peut etre NULL, un site sans capacite declaree
n'est pas une erreur, seul le calcul du seuil s'en trouve prive de repere."""


def load_sites(connection: ConnectionLike) -> Sequence[SiteReference]:
    """Charge le referentiel des sites connus de la base.

    Args:
        connection: Connexion ouverte vers la base.

    Returns:
        Un SiteReference par site, capacity_kw absent si non renseigne en base.

    Raises:
        DatabaseQueryError: Si le pilote echoue a executer la requete.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(SELECT_SITE_CATALOG)
            rows = cursor.fetchall()
    except Exception as failure:
        raise DatabaseQueryError("site catalog", str(failure)) from failure

    return [
        SiteReference(site_id=str(row[0]), capacity_kw=_as_optional_int(row[1])) for row in rows
    ]


def _as_optional_int(value: object) -> Optional[int]:
    return None if value is None else int(value)  # type: ignore[call-overload]
