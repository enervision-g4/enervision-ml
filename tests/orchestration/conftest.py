from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Optional

from enervision_ml.records import Observation, SiteReference


class FakeCursor:
    """Curseur en memoire qui enregistre les insertions du run."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, statement: str, parameters: Optional[Any] = None) -> None:
        self.statements.append(statement)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return []

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        return None

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
    """Connexion en memoire : compte les commits et les rollbacks du run."""

    def __init__(self) -> None:
        self.opened_cursor = FakeCursor()
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> FakeCursor:
        return self.opened_cursor

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        pass


class StubEstimator:
    """Estimateur factice : predit une valeur fixe, sans aucun calcul reel."""

    def __init__(self, fixed_prediction: float = 15.0) -> None:
        self.fixed_prediction = fixed_prediction

    def fit(self, features_matrix: Any, target: Any) -> None:
        pass

    def predict(self, features_matrix: Any) -> list[float]:
        return [self.fixed_prediction for _ in features_matrix]


class StubHistorySource:
    """Source d'historique factice : sert des observations preparees en memoire."""

    def __init__(
        self,
        observations_by_site: dict[str, list[Observation]],
        sites: Optional[list[SiteReference]] = None,
    ) -> None:
        self._observations_by_site = observations_by_site
        self._sites = sites if sites is not None else [
            SiteReference(site_id=site_id, capacity_kw=None) for site_id in observations_by_site
        ]

    def load_observations(self, site_id: str) -> list[Observation]:
        return self._observations_by_site.get(site_id, [])

    def load_site_catalog(self) -> list[SiteReference]:
        return self._sites


class FakeMlflowRun:
    """Run MLflow factice : porte juste un identifiant pour verifier l'imbrication."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class FakeMlflowClient:
    """Client MLflow factice : enregistre chaque appel, sans reseau reel.

    raise_on permet de simuler l'echec d'une methode precise (site en panne de
    tracking), pour verifier que log_evaluation_report ne laisse jamais un run
    ouvert ni ne remonte l'erreur.
    """

    def __init__(self, raise_on: Optional[str] = None) -> None:
        self.raise_on = raise_on
        self.started_runs: list[tuple[Optional[str], bool]] = []
        self.logged_params: list[tuple[str, object]] = []
        self.logged_metrics: list[tuple[str, float]] = []
        self.ended_run_count = 0
        self._next_run_id = 0

    def _maybe_raise(self, method_name: str) -> None:
        if self.raise_on == method_name:
            raise RuntimeError(f"{method_name} failed")

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> FakeMlflowRun:
        self._maybe_raise("start_run")
        self.started_runs.append((run_name, nested))
        self._next_run_id += 1
        return FakeMlflowRun(run_id=f"run-{self._next_run_id}")

    def log_param(self, key: str, value: object) -> None:
        self._maybe_raise("log_param")
        self.logged_params.append((key, value))

    def log_metric(self, key: str, value: float) -> None:
        self._maybe_raise("log_metric")
        self.logged_metrics.append((key, value))

    def end_run(self) -> None:
        self._maybe_raise("end_run")
        self.ended_run_count += 1


class FakeWarningLogger:
    """Journal factice : n'enregistre que les avertissements, pour verifier
    qu'un echec de tracking est bien signale sans faire planter l'appelant."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **kw: object) -> None:
        self.warnings.append((event, kw))


def make_hourly_observations(
    site_id: str, count: int, consumption_kw: float = 10.0
) -> list[Observation]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Observation(
            site_id=site_id,
            timestamp=start + timedelta(hours=index),
            consumption_kw=consumption_kw + (index % 5),
            temperature_celsius=18.0,
            humidity_percent=60.0,
        )
        for index in range(count)
    ]
