"""Exceptions de la couche de transformation.

Uniquement des echecs de contrat sur des donnees deja extraites : une colonne absente
ou un historique trop court, jamais une panne d'E/S, qui reste du ressort de extract/.
"""


class FeatureError(Exception):
    """Classe de base des erreurs levees lors de la construction des features."""


class NotEnoughHistoryError(FeatureError):
    """L'historique disponible est trop court pour entrainer un modele.

    Attributes:
        site_id: Site dont l'historique est insuffisant.
        available_hours: Nombre d'heures d'observations disponibles.
        required_hours: Nombre d'heures minimal exige.
    """

    def __init__(self, site_id: str, available_hours: int, required_hours: int) -> None:
        """Construit l'erreur pour un site et une couverture donnes.

        Args:
            site_id: Site dont l'historique est insuffisant.
            available_hours: Nombre d'heures d'observations disponibles.
            required_hours: Nombre d'heures minimal exige.
        """
        super().__init__(
            f"site {site_id!r} has only {available_hours} h of history, "
            f"{required_hours} h are required to train"
        )
        self.site_id = site_id
        self.available_hours = available_hours
        self.required_hours = required_hours
