"""Construction des lignes de prevision, aucun effet de bord.

L'identifiant de chaque prevision est deterministe (uuid5) : rejouer un lot produit
exactement les memes identifiants, ce qui permet a l'insertion ON CONFLICT DO NOTHING
d'absorber le doublon plutot que de dupliquer la ligne.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from ..records import PredictionRow


def build_target_timestamps(now: datetime, horizon_hours: int) -> list[datetime]:
    """Construit les heures pleines visees par un lot de prevision.

    Args:
        now: Instant de generation du lot.
        horizon_hours: Nombre d'heures a couvrir.

    Returns:
        Les horodatages, en UTC, a partir de la prochaine heure pleine.
    """
    next_whole_hour = now.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    ) + timedelta(hours=1)
    return [next_whole_hour + timedelta(hours=offset) for offset in range(horizon_hours)]


def build_prediction_id(site_id: str, target_timestamp: datetime, model_version: str) -> UUID:
    """Derive un identifiant deterministe de prevision.

    Args:
        site_id: Site concerne.
        target_timestamp: Heure visee par la prevision.
        model_version: Empreinte du contrat de modele utilise.

    Returns:
        Un UUID5, identique pour un meme triplet (site, heure visee, version).
    """
    payload = f"{site_id}|{target_timestamp.isoformat()}|{model_version}"
    return uuid5(NAMESPACE_URL, payload)


def build_prediction_rows(
    site_id: str,
    target_timestamps: Sequence[datetime],
    predicted_consumption_kw: Sequence[float],
    threshold_kw: Optional[float],
    model_version: str,
    generated_at: datetime,
) -> list[PredictionRow]:
    """Assemble les lignes de prevision d'un lot.

    Args:
        site_id: Site concerne.
        target_timestamps: Heures visees, dans le meme ordre que predicted_consumption_kw.
        predicted_consumption_kw: Puissances predites, dans le meme ordre.
        threshold_kw: Seuil d'alerte du site au moment du run.
        model_version: Empreinte du contrat de modele utilise.
        generated_at: Instant de generation, commun a toutes les lignes du lot.

    Returns:
        Une ligne par heure visee.
    """
    return [
        PredictionRow(
            prediction_id=build_prediction_id(site_id, target_timestamp, model_version),
            site_id=site_id,
            target_timestamp=target_timestamp,
            predicted_consumption_kw=predicted_value,
            threshold_kw=threshold_kw,
            model_version=model_version,
            timestamp=generated_at,
        )
        for target_timestamp, predicted_value in zip(
            target_timestamps, predicted_consumption_kw, strict=True
        )
    ]
