from datetime import UTC, datetime, timedelta
from typing import Any, Optional

from enervision_ml.records import Observation


class StubEstimator:
    """Estimateur factice : enregistre ses appels, ne fait aucun calcul reel."""

    def __init__(self, fixed_prediction: float = 42.0) -> None:
        self.fixed_prediction = fixed_prediction
        self.fit_calls: list[tuple[Any, Any]] = []
        self.predict_calls: list[Any] = []

    def fit(self, features_matrix: Any, target: Any) -> None:
        self.fit_calls.append((features_matrix, target))

    def predict(self, features_matrix: Any) -> list[float]:
        self.predict_calls.append(features_matrix)
        return [self.fixed_prediction for _ in features_matrix]


def make_observations(count: int, consumption_kw: Optional[float] = 10.0) -> list[Observation]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Observation(
            site_id="SITE001",
            timestamp=start + timedelta(hours=index),
            consumption_kw=consumption_kw,
            temperature_celsius=18.0,
            humidity_percent=60.0,
        )
        for index in range(count)
    ]
