"""Construction de la ligne de features consommee par l'estimateur.

L'intersection stricte des deux sources d'historique : le CSV du formateur et
measure_imputed. solar_irradiance_wm2 existe cote CSV mais pas cote base, elle est
donc portee par la structure sans figurer ici, voir weather_outlook.py.
"""

import math
from datetime import UTC
from typing import Optional

from ..records import Observation

FEATURE_NAMES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "month",
    "temperature_celsius",
    "humidity_percent",
)
"""Ordre de colonne du contrat de features. Son empreinte alimente build_model_version."""

WEEKEND_WEEKDAYS = frozenset({5, 6})
"""Samedi et dimanche, au sens de datetime.weekday() (lundi=0)."""


def build_feature_row(observation: Observation) -> tuple[float, ...]:
    """Derive une ligne de features d'une observation horaire.

    Le calendaire est recalcule depuis l'instant UTC de l'observation, jamais lu
    d'une colonne precalculee : celle-ci est probablement en heure locale, ce qui
    entrainerait le modele sur un calendrier different de celui de la production.

    Args:
        observation: Mesure horaire dont deriver les features.

    Returns:
        Les valeurs de FEATURE_NAMES, dans le meme ordre. Une grandeur absente vaut
        NaN plutot que zero, que l'estimateur distingue nativement d'une vraie mesure.
    """
    instant = observation.timestamp.astimezone(UTC)
    day_of_week = instant.weekday()

    return (
        float(instant.hour),
        float(day_of_week),
        float(day_of_week in WEEKEND_WEEKDAYS),
        float(instant.month),
        _or_nan(observation.temperature_celsius),
        _or_nan(observation.humidity_percent),
    )


def _or_nan(value: Optional[float]) -> float:
    return math.nan if value is None else value
