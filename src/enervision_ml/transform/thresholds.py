"""Calcul du seuil d'alerte associe a une prevision.

capacity_kw * ratio quand la capacite du site est connue. A defaut, le percentile
haut de l'historique d'entrainement sert de repere observe, plutot qu'un seuil
invente ; en dernier recours, sans aucun historique, aucun seuil n'est ecrit.
"""

from collections.abc import Sequence
from typing import Optional

HIGH_PERCENTILE = 0.95
"""Repli quand la capacite installee est inconnue : la fraction haute de l'historique."""

DEFAULT_CAPACITY_RATIO = 0.85
"""Fraction de la capacite installee retenue comme seuil d'alerte par defaut."""


def compute_threshold_kw(
    capacity_kw: Optional[int],
    observed_consumption_kw: Sequence[float],
    ratio: float = DEFAULT_CAPACITY_RATIO,
) -> Optional[float]:
    """Calcule le seuil d'alerte d'un site.

    Args:
        capacity_kw: Puissance installee du site, si connue du referentiel.
        observed_consumption_kw: Consommations connues de l'historique d'entrainement,
            utilisees en repli si la capacite est inconnue.
        ratio: Fraction de la capacite retenue comme seuil.

    Returns:
        capacity_kw * ratio si la capacite est connue ; sinon le percentile haut de
        observed_consumption_kw ; sinon None si l'historique est vide.
    """
    if capacity_kw is not None:
        return capacity_kw * ratio
    if not observed_consumption_kw:
        return None
    return _percentile(sorted(observed_consumption_kw), HIGH_PERCENTILE)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = rank - lower_index
    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
