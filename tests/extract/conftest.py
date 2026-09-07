from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Optional


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeCursor:
    """Curseur en memoire qui enregistre ce que la source envoie a la base.

    Il ne simule pas Postgres : les semantiques d'agregation sont verifiees contre une
    vraie base, pas ici. Il sert a observer la requete emise et a scenariser une
    erreur du pilote.
    """

    def __init__(
        self,
        rows: Optional[Sequence[tuple[Any, ...]]] = None,
        failure: Optional[Exception] = None,
    ) -> None:
        self.statements: list[str] = []
        self.parameters: list[Sequence[Any]] = []
        self._rows = rows if rows is not None else []
        self._failure = failure

    def execute(self, statement: str, parameters: Optional[Sequence[Any]] = None) -> None:
        self.statements.append(statement)
        self.parameters.append(parameters if parameters is not None else ())
        if self._failure is not None:
            raise self._failure

    def fetchall(self) -> Sequence[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        return self._rows[0] if self._rows else None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        exception_traceback: Optional[TracebackType],
    ) -> None:
        return None


class FakeConnection:
    """Connexion en memoire, qui rend toujours le meme curseur observable."""

    def __init__(
        self,
        rows: Optional[Sequence[tuple[Any, ...]]] = None,
        failure: Optional[Exception] = None,
    ) -> None:
        self.opened_cursor = FakeCursor(rows, failure)
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.opened_cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True
