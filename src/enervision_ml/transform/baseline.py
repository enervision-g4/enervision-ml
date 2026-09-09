"""Profil horaire moyen : la reference a laquelle comparer le modele.

Jamais ecrit en base, ce profil sert uniquement a la commande evaluate a prouver que
l'estimateur apprend plus qu'un moyennage naif par heure de la journee.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional

from ..records import Observation


@dataclass(frozen=True)
class HourlyProfile:
    """Profil horaire moyen, ajuste sur un historique d'entrainement.

    Attributes:
        mean_by_hour: Puissance moyenne connue pour chaque heure UTC de la journee
            (0 a 23) rencontree dans l'historique d'entrainement.
    """

    mean_by_hour: Mapping[int, float]


def fit_hourly_profile(observations: Sequence[Observation]) -> HourlyProfile:
    """Calcule le profil horaire moyen d'un site.

    Args:
        observations: Historique d'entrainement du site.

    Returns:
        Le profil horaire moyen, sur les seules heures UTC ayant au moins une
        consommation connue.

    Raises:
        ValueError: Si aucune observation ne porte de consommation connue.
    """
    values_by_hour: dict[int, list[float]] = defaultdict(list)
    for observation in observations:
        if observation.consumption_kw is not None:
            hour = observation.timestamp.astimezone(UTC).hour
            values_by_hour[hour].append(observation.consumption_kw)

    if not values_by_hour:
        raise ValueError("cannot fit an hourly profile without any known consumption value")

    return HourlyProfile(
        mean_by_hour={hour: sum(values) / len(values) for hour, values in values_by_hour.items()}
    )


def predict_from_profile(profile: HourlyProfile, timestamp: datetime) -> Optional[float]:
    """Predit une puissance a partir du profil horaire moyen.

    Args:
        profile: Profil ajuste par fit_hourly_profile.
        timestamp: Instant vise.

    Returns:
        La moyenne connue pour l'heure UTC de l'instant, ou None si cette heure n'a
        jamais ete observee dans l'historique d'entrainement.
    """
    return profile.mean_by_hour.get(timestamp.astimezone(UTC).hour)
