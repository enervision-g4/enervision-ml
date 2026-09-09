import os
import signal
import sys
from collections.abc import Iterator

import pytest

from enervision_ml.orchestration.graceful_shutdown import ShutdownRequest

# Sur Windows, os.kill(pid, SIGTERM/SIGINT) invoque TerminateProcess plutot que le
# gestionnaire Python enregistre : le processus pytest est tue net, sans passer par
# ShutdownRequest.request(). Le conteneur de production reste Linux (voir Dockerfile),
# ces tests y sont donc pertinents ; ils sont juste dangereux a executer localement
# sous Windows, ou ils interrompraient toute la suite sans le moindre message d'erreur.
REAL_SIGNALS_KILL_THE_PROCESS_ON_WINDOWS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.kill(pid, SIGTERM/SIGINT) tue le processus sous Windows au lieu "
    "d'invoquer le gestionnaire ; le comportement reel est verifie en conteneur Linux",
)


@pytest.fixture
def restored_handlers() -> Iterator[None]:
    """Restaure les gestionnaires de signaux, qui sont un etat global du processus."""
    previous = {
        received: signal.getsignal(received) for received in (signal.SIGTERM, signal.SIGINT)
    }
    yield
    for received, handler in previous.items():
        signal.signal(received, handler)


def test_no_shutdown_is_requested_at_startup() -> None:
    assert ShutdownRequest().requested is False


def test_requesting_a_shutdown_raises_the_flag() -> None:
    request = ShutdownRequest()

    request.request()

    assert request.requested is True


def test_installing_replaces_the_default_handlers(restored_handlers: None) -> None:
    request = ShutdownRequest()

    request.install()

    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_DFL


@REAL_SIGNALS_KILL_THE_PROCESS_ON_WINDOWS
def test_a_real_sigterm_raises_the_flag_instead_of_killing_the_process(
    restored_handlers: None,
) -> None:
    # Sans gestionnaire, SIGTERM interrompt le processus sans executer les blocs
    # finally : un lot de prevision en cours d'ecriture serait interrompu en plein site.
    request = ShutdownRequest()
    request.install()

    os.kill(os.getpid(), signal.SIGTERM)

    assert request.requested is True


@REAL_SIGNALS_KILL_THE_PROCESS_ON_WINDOWS
def test_a_real_sigint_raises_the_flag(restored_handlers: None) -> None:
    request = ShutdownRequest()
    request.install()

    os.kill(os.getpid(), signal.SIGINT)

    assert request.requested is True


@REAL_SIGNALS_KILL_THE_PROCESS_ON_WINDOWS
def test_a_second_signal_leaves_the_flag_raised(restored_handlers: None) -> None:
    request = ShutdownRequest()
    request.install()

    os.kill(os.getpid(), signal.SIGTERM)
    os.kill(os.getpid(), signal.SIGTERM)

    assert request.requested is True
