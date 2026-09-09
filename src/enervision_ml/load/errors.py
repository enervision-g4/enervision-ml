"""Erreurs de la couche de chargement."""


class PersistenceError(Exception):
    """Une ecriture en base n'a pas abouti.

    Attributes:
        table: Table dans laquelle l'ecriture a echoue.
        reason: Description technique rendue par le pilote.
    """

    def __init__(self, table: str, reason: str) -> None:
        """Rassemble la table visee et la cause du refus.

        Args:
            table: Table dans laquelle l'ecriture a echoue.
            reason: Description technique rendue par le pilote.
        """
        super().__init__(f"write to table {table!r} failed: {reason}")
        self.table = table
        self.reason = reason


class UnknownSiteReferenceError(PersistenceError):
    """Une prevision reference un site absent du referentiel.

    Attributes:
        site_id: Identifiant du site absent du referentiel.
    """

    def __init__(self, table: str, site_id: str) -> None:
        """Rassemble la table visee et le site introuvable.

        Args:
            table: Table dans laquelle l'ecriture a echoue.
            site_id: Identifiant du site absent du referentiel.
        """
        super().__init__(table, f"site {site_id!r} is not in the registry")
        self.site_id = site_id
