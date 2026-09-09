"""Exceptions du sous-paquet modele.

Distinctes des erreurs de transformation : un estimateur est un objet a etat, ses
echecs sont ceux d'un cycle de vie (pas assez d'exemples, predire avant d'entrainer),
pas ceux d'un contrat de donnees deja extraites.
"""


class TrainingError(Exception):
    """Classe de base des erreurs du sous-paquet modele."""


class NotEnoughSamplesError(TrainingError):
    """Trop peu d'exemples exploitables pour entrainer l'estimateur.

    Attributes:
        site_id: Site concerne.
        sample_count: Nombre d'exemples exploitables, c'est-a-dire de consommation connue.
        minimum_required: Nombre minimal exige.
    """

    def __init__(self, site_id: str, sample_count: int, minimum_required: int) -> None:
        """Construit l'erreur pour un site et un compte d'exemples donnes.

        Args:
            site_id: Site concerne.
            sample_count: Nombre d'exemples exploitables.
            minimum_required: Nombre minimal exige.
        """
        super().__init__(
            f"site {site_id!r} has only {sample_count} usable samples, "
            f"{minimum_required} are required to train"
        )
        self.site_id = site_id
        self.sample_count = sample_count
        self.minimum_required = minimum_required


class UnfittedModelError(TrainingError):
    """predict() est appele avant fit().

    Attributes:
        site_id: Site dont le modele n'est pas encore entraine.
    """

    def __init__(self, site_id: str) -> None:
        """Construit l'erreur pour un site dont le modele n'est pas encore entraine.

        Args:
            site_id: Site dont le modele n'est pas encore entraine.
        """
        super().__init__(f"model for site {site_id!r} was not fitted before predict()")
        self.site_id = site_id
