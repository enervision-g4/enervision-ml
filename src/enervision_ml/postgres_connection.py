"""Acces a la base, reduit au sous-ensemble du pilote reellement utilise.

A la racine du paquet, pas dans extract/ ni dans load/ : une meme connexion sert la
lecture de l'historique et l'ecriture des previsions dans une seule transaction par
site, voir orchestration/forecast_run.py.
"""

from collections.abc import Sequence
from types import TracebackType
from typing import Optional, Protocol, cast


class CursorLike(Protocol):
    """Sous-ensemble du curseur psycopg reellement utilise ici."""

    def execute(self, statement: str, parameters: Optional[Sequence[object]] = None) -> None:
        """Envoie une requete parametree a la base."""
        ...

    def fetchone(self) -> Optional[tuple[object, ...]]:
        """Rend la premiere ligne du resultat, ou None si la requete n'en a produit aucune."""
        ...

    def fetchall(self) -> Sequence[tuple[object, ...]]:
        """Rend toutes les lignes du resultat, eventuellement une sequence vide."""
        ...

    def __enter__(self) -> "CursorLike":
        """Ouvre le bloc de contexte du curseur."""
        ...

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[TracebackType],
    ) -> None:
        """Referme le curseur a la sortie du bloc."""
        ...


class ConnectionLike(Protocol):
    """Sous-ensemble de la connexion psycopg reellement utilisee ici."""

    def cursor(self) -> CursorLike:
        """Ouvre un curseur sur cette connexion."""
        ...

    def commit(self) -> None:
        """Valide la transaction en cours."""
        ...

    def rollback(self) -> None:
        """Annule la transaction en cours."""
        ...

    def close(self) -> None:
        """Ferme la connexion."""
        ...


def create_connection(database_url: str) -> ConnectionLike:
    """Ouvre une connexion a la base.

    Args:
        database_url: URL de connexion, schema postgres:// ou postgresql://.

    Returns:
        La connexion, en validation manuelle : un site en echec peut ainsi etre
        annule sans affecter les sites deja ecrits dans le meme run.

    Raises:
        ValueError: Si aucune URL n'est fournie.
    """
    if not database_url:
        raise ValueError("database_url is required to reach the database")

    import psycopg

    return cast(ConnectionLike, psycopg.connect(database_url, autocommit=False))
