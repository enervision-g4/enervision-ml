"""Vocabulaire commun du service.

Des enregistrements immuables plutot que des dictionnaires : un champ renomme ou
supprime se detecte a la lecture du code, pas a l'execution d'un traitement en aval.
Ce module ne depend que de la bibliotheque standard, comme le reste du paquet.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID


@dataclass(frozen=True)
class Observation:
    """Une mesure horaire de consommation, brute ou reconstruite selon la source.

    Attributes:
        site_id: Identifiant du site.
        timestamp: Debut de l'heure observee, en UTC.
        consumption_kw: Puissance moyenne sur l'heure, absente si aucune mesure valide.
        temperature_celsius: Temperature moyenne sur l'heure, absente si le capteur a manque.
        humidity_percent: Humidite moyenne sur l'heure, absente si le capteur a manque.
    """

    site_id: str
    timestamp: datetime
    consumption_kw: Optional[float]
    temperature_celsius: Optional[float]
    humidity_percent: Optional[float]


@dataclass(frozen=True)
class SiteReference:
    """Caracteristiques statiques d'un site, utiles au calcul du seuil d'alerte.

    Attributes:
        site_id: Identifiant du site.
        capacity_kw: Puissance installee, absente si le site n'est pas dans le referentiel.
    """

    site_id: str
    capacity_kw: Optional[int]


@dataclass(frozen=True)
class PredictionRow:
    """Une prevision destinee a la table prediction.

    Attributes:
        prediction_id: Identifiant deterministe de la prevision, voir build_prediction_id.
        site_id: Site concerne.
        target_timestamp: Heure visee par la prevision.
        predicted_consumption_kw: Puissance prevue.
        threshold_kw: Seuil d'alerte du site au moment de la prevision, absent si inconnu.
        model_version: Empreinte du contrat de features et de l'estimateur.
        timestamp: Instant de generation du lot, commun a toutes les previsions du run.
    """

    prediction_id: UUID
    site_id: str
    target_timestamp: datetime
    predicted_consumption_kw: float
    threshold_kw: Optional[float]
    model_version: str
    timestamp: datetime


@dataclass(frozen=True)
class RecommendationRow:
    """Une recommandation destinee a la table recommendation.

    Attributes:
        recommendation_id: Identifiant de la recommandation.
        site_id: Site concerne.
        prediction_id: Prevision a l'origine de la recommandation.
        timestamp: Instant de generation.
        action_description: Description de l'action suggeree.
        status: Etat de la recommandation.
    """

    recommendation_id: UUID
    site_id: str
    prediction_id: UUID
    timestamp: datetime
    action_description: str
    status: str
