"""Recommandations derivees des previsions qui depassent le seuil d'alerte.

Une recommandation par prevision dont la puissance predite depasse strictement son
seuil. Elle n'est jamais ecrite sans sa prevision : les deux sont commit dans la meme
transaction par site, voir orchestration/forecast_run.py.
"""

from collections.abc import Sequence
from uuid import uuid4

from ..records import PredictionRow, RecommendationRow

RECOMMENDATION_STATUS = "open"
"""Etat initial d'une recommandation, avant tout traitement humain."""


def build_recommendations(predictions: Sequence[PredictionRow]) -> list[RecommendationRow]:
    """Derive une recommandation pour chaque prevision qui depasse son seuil.

    Args:
        predictions: Previsions d'un lot, avec ou sans seuil connu.

    Returns:
        Une recommandation par prevision dont predicted_consumption_kw depasse
        strictement threshold_kw. Aucune n'est generee pour un seuil inconnu (None) :
        il n'y a rien a comparer, ni pour une prevision qui atteint son seuil sans le
        depasser.
    """
    recommendations: list[RecommendationRow] = []
    for prediction in predictions:
        if prediction.threshold_kw is None:
            continue
        if prediction.predicted_consumption_kw <= prediction.threshold_kw:
            continue
        recommendations.append(_build_recommendation(prediction, prediction.threshold_kw))
    return recommendations


def _build_recommendation(prediction: PredictionRow, threshold_kw: float) -> RecommendationRow:
    hour_label = prediction.target_timestamp.strftime("%Y-%m-%d %H:%M")
    action_description = (
        f"Consommation prevue de {prediction.predicted_consumption_kw:.1f} kW a "
        f"{hour_label} UTC, au-dessus du seuil de {threshold_kw:.1f} kW."
    )
    return RecommendationRow(
        recommendation_id=uuid4(),
        site_id=prediction.site_id,
        prediction_id=prediction.prediction_id,
        timestamp=prediction.timestamp,
        action_description=action_description,
        status=RECOMMENDATION_STATUS,
    )
