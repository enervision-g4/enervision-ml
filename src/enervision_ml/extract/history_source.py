"""Contrat commun aux sources d'historique, CSV ou base.

Un Protocol plutot qu'une classe abstraite : csv_history.py et database_history.py
n'ont pas besoin d'heriter d'un ancetre commun, le contrat suffit, et les doubles de
test n'ont rien a importer de concret pour le satisfaire.
"""

from collections.abc import Sequence
from typing import Protocol

from ..records import Observation, SiteReference


class HistorySourceLike(Protocol):
    """Source d'observations horaires et de referentiel de sites."""

    def load_observations(self, site_id: str) -> Sequence[Observation]:
        """Charge l'historique horaire d'un site.

        Args:
            site_id: Site dont l'historique est demande.

        Returns:
            Les observations disponibles, triees par instant croissant.
        """
        ...

    def load_site_catalog(self) -> Sequence[SiteReference]:
        """Charge le referentiel des sites connus de la source.

        Returns:
            Les sites, dans un ordre quelconque.
        """
        ...
