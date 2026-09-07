from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from enervision_ml.records import Observation, SiteReference


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
