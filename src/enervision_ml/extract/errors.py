"""Exceptions de la couche d'extraction.

Une panne de lecture (colonne absente, ligne malformee, pilote indisponible) est une
erreur d'exploitation ou de configuration, jamais un defaut de contrat sur des donnees
deja extraites : ces dernieres relevent de transform/errors.py.
"""


class ExtractionError(Exception):
    """Classe de base de toutes les erreurs remontees par la couche d'extraction."""


class MissingColumnError(ExtractionError):
    """Une colonne obligatoire est absente de la source lue.

    Attributes:
        column_name: Nom de la colonne manquante.
        source_path: Origine lue, pour retrouver le fichier fautif.
    """

    def __init__(self, column_name: str, source_path: str) -> None:
        """Construit l'erreur pour une colonne et une source donnees.

        Args:
            column_name: Nom de la colonne manquante.
            source_path: Origine lue, pour retrouver le fichier fautif.
        """
        super().__init__(f"column {column_name!r} is missing from {source_path!r}")
        self.column_name = column_name
        self.source_path = source_path


class MalformedRowError(ExtractionError):
    """Une ligne ne peut pas etre interpretee selon le contrat attendu.

    Attributes:
        row_number: Numero de la ligne fautive, en comptant l'entete comme ligne 1.
        reason: Description de l'echec de conversion.
    """

    def __init__(self, row_number: int, reason: str) -> None:
        """Construit l'erreur pour une ligne et un motif donnes.

        Args:
            row_number: Numero de la ligne fautive, en comptant l'entete comme ligne 1.
            reason: Description de l'echec de conversion.
        """
        super().__init__(f"row {row_number} is malformed: {reason}")
        self.row_number = row_number
        self.reason = reason


class InconsistentSamplingError(ExtractionError):
    """Le pas d'echantillonnage d'un site n'est pas exploitable.

    Attributes:
        site_id: Site dont le pas est en cause.
        detail: Description de l'anomalie observee.
    """

    def __init__(self, site_id: str, detail: str) -> None:
        """Construit l'erreur pour un site et une anomalie donnes.

        Args:
            site_id: Site dont le pas est en cause.
            detail: Description de l'anomalie observee.
        """
        super().__init__(f"sampling step for site {site_id!r} is not usable: {detail}")
        self.site_id = site_id
        self.detail = detail


class DatabaseQueryError(ExtractionError):
    """Le pilote echoue a executer une requete de lecture.

    Attributes:
        context: Ce qui etait interroge, un site ou "site catalog".
        cause: Message technique renvoye par le pilote.
    """

    def __init__(self, context: str, cause: str) -> None:
        """Construit l'erreur pour une requete et une cause donnees.

        Args:
            context: Ce qui etait interroge, un site ou "site catalog".
            cause: Message technique renvoye par le pilote.
        """
        super().__init__(f"database query failed for {context!r}: {cause}")
        self.context = context
        self.cause = cause
