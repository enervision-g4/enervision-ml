"""Projection climatologique de la meteo, faute de prevision meteo disponible.

Predire H+24 demande la temperature de H+24, que personne ne fournit : elle est
remplacee par la moyenne climatologique (mois, heure) de l'historique du site. Le MAE
affiche par evaluate reste donc optimiste, puisqu'il utilise la temperature observee ;
c'est un ecart assume, documente dans le README plutot qu'enfoui dans le code.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional

from ..records import Observation


@dataclass(frozen=True)
class WeatherPoint:
    """Meteo projetee pour un instant futur.

    Attributes:
        temperature_celsius: Temperature climatologique (mois, heure), absente si ce
            couple n'a jamais ete observe.
        humidity_percent: Humidite climatologique (mois, heure), absente si ce couple
            n'a jamais ete observe.
    """

    temperature_celsius: Optional[float]
    humidity_percent: Optional[float]


@dataclass(frozen=True)
class Climatology:
    """Moyenne (mois, heure UTC) de la temperature et de l'humidite d'un site.

    Attributes:
        temperature_by_month_hour: Temperature moyenne connue par couple (mois, heure).
        humidity_by_month_hour: Humidite moyenne connue par couple (mois, heure).
    """

    temperature_by_month_hour: Mapping[tuple[int, int], float]
    humidity_by_month_hour: Mapping[tuple[int, int], float]


def build_climatology(observations: Sequence[Observation]) -> Climatology:
    """Calcule la moyenne climatologique (mois, heure) d'un site.

    Args:
        observations: Historique du site.

    Returns:
        La climatologie, sur les seuls couples (mois, heure) ayant au moins une
        valeur connue ; chaque grandeur est moyennee independamment de l'autre.
    """
    temperature_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    humidity_values: dict[tuple[int, int], list[float]] = defaultdict(list)

    for observation in observations:
        instant = observation.timestamp.astimezone(UTC)
        key = (instant.month, instant.hour)
        if observation.temperature_celsius is not None:
            temperature_values[key].append(observation.temperature_celsius)
        if observation.humidity_percent is not None:
            humidity_values[key].append(observation.humidity_percent)

    return Climatology(
        temperature_by_month_hour={
            key: sum(values) / len(values) for key, values in temperature_values.items()
        },
        humidity_by_month_hour={
            key: sum(values) / len(values) for key, values in humidity_values.items()
        },
    )


def project_weather(climatology: Climatology, target_timestamp: datetime) -> WeatherPoint:
    """Projete une meteo plausible pour un instant futur.

    Args:
        climatology: Climatologie ajustee par build_climatology.
        target_timestamp: Instant vise.

    Returns:
        La moyenne climatologique du couple (mois, heure UTC) de l'instant vise,
        absente si ce couple n'a jamais ete observe.
    """
    instant = target_timestamp.astimezone(UTC)
    key = (instant.month, instant.hour)
    return WeatherPoint(
        temperature_celsius=climatology.temperature_by_month_hour.get(key),
        humidity_percent=climatology.humidity_by_month_hour.get(key),
    )
