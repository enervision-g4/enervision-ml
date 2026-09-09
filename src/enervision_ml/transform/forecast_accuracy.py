"""Justesse d'une prevision desormais resolue, pour le suivi MLflow de forecast.

Compare la derniere prevision ecrite pour l'heure la plus recente desormais connue a la
mesure reelle qui vient d'arriver, plutot qu'a un decoupage train/test artificiel (voir
transform/evaluation.py, reserve a evaluate). Le signal porte ainsi sur le modele tel
qu'il tourne reellement en production, avec le delai d'une heure necessaire pour que la
verite soit connue.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from ..records import Observation
from .evaluation import MapeResult, mean_absolute_percentage_error


@dataclass(frozen=True)
class ForecastAccuracy:
    """Justesse d'une prevision resolue, prete a journaliser.

    Attributes:
        mae: Ecart absolu entre la prevision et la mesure reelle, en kW.
        mape: Ecart relatif, absent si la mesure reelle etait trop proche de zero
            (voir NEAR_ZERO_TRUTH_KW dans transform/evaluation.py).
    """

    mae: float
    mape: Optional[MapeResult]


def most_recently_resolved_observation(
    observations: Sequence[Observation],
) -> Optional[Observation]:
    """Trouve l'observation la plus recente d'un historique deja charge.

    Args:
        observations: Historique horaire du site. ForecastRun.run() ne conserve deja
            que les observations dont consumption_kw est connu.

    Returns:
        L'observation la plus recente, ou None si l'historique est vide.
    """
    return max(observations, key=lambda observation: observation.timestamp, default=None)


def compute_forecast_accuracy(
    predicted_consumption_kw: float, actual_consumption_kw: float
) -> ForecastAccuracy:
    """Compare une prevision resolue a la mesure reelle desormais connue.

    Args:
        predicted_consumption_kw: Ce que le modele avait predit pour cette heure.
        actual_consumption_kw: Ce qui a reellement ete mesure.

    Returns:
        L'ecart absolu, et l'ecart relatif quand la mesure reelle le permet.
    """
    mae = abs(actual_consumption_kw - predicted_consumption_kw)
    try:
        mape = mean_absolute_percentage_error(
            [actual_consumption_kw], [predicted_consumption_kw]
        )
    except ValueError:
        # Mesure reelle trop proche de zero (voir NEAR_ZERO_TRUTH_KW) : un ecart
        # relatif exploserait pour un ecart absolu negligeable, mape reste absent.
        mape = None
    return ForecastAccuracy(mae=mae, mape=mape)
