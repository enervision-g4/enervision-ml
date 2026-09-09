from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Empeche la configuration de journalisation d'un test de fuir dans les suivants.

    configure_logging retient le flux de sortie qu'on lui donne, et cette configuration
    est globale au processus. Tout test qui lance une commande, ou qui capture stderr,
    laisserait sinon derriere lui un flux que pytest refermera, et les tests suivants
    ecriraient dans un fichier ferme.
    """
    yield
    structlog.reset_defaults()
