"""Regroupement des observations au pas horaire, en kW, jamais en somme de kWh.

Une moyenne de puissance reste une puissance correcte meme quand une heure n'a que
quelques mesures ; une somme donnerait un creux fantome que le modele apprendrait.
Ce module est agnostique de la source : il recoit des Observation deja converties en
kW, et n'est jamais appele du tout par l'extracteur base, qui delegue l'agregation a
PostgreSQL.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Optional

from ..records import Observation


def infer_sampling_step(timestamps: Sequence[datetime]) -> timedelta:
    """Determine le pas d'echantillonnage source.

    La mediane des ecarts est retenue plutot que le premier ecart observe : un seul
    intervalle irregulier en tete de serie (retard de collecte, resynchronisation) ne
    doit pas fausser la conversion kWh -> kW qui depend de ce pas.

    Args:
        timestamps: Horodatages d'un meme site, dans un ordre quelconque.

    Returns:
        Le pas median entre deux horodatages consecutifs, une fois tries.

    Raises:
        ValueError: Si moins de deux horodatages sont fournis.
    """
    if len(timestamps) < 2:
        raise ValueError("at least two timestamps are required to infer a sampling step")
    ordered = sorted(timestamps)
    gaps = sorted(later - earlier for earlier, later in pairwise(ordered))
    return gaps[len(gaps) // 2]


def aggregate_to_hourly(samples: Sequence[Observation]) -> list[Observation]:
    """Regroupe des observations par heure UTC pleine et par site.

    Chaque grandeur est la moyenne des valeurs connues du godet horaire. Un godet
    sans aucune valeur connue reste absent, jamais ramene a zero.

    Args:
        samples: Observations a regrouper, a un pas quelconque.

    Returns:
        Une observation par site et par heure couverte, triees par site puis par
        instant croissant.
    """
    buckets: dict[tuple[str, datetime], list[Observation]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.site_id, _bucket_start(sample.timestamp))].append(sample)

    aggregated = [
        Observation(
            site_id=site_id,
            timestamp=bucket_start,
            consumption_kw=_mean_or_none(s.consumption_kw for s in bucket_samples),
            temperature_celsius=_mean_or_none(s.temperature_celsius for s in bucket_samples),
            humidity_percent=_mean_or_none(s.humidity_percent for s in bucket_samples),
        )
        for (site_id, bucket_start), bucket_samples in buckets.items()
    ]
    return sorted(aggregated, key=lambda observation: (observation.site_id, observation.timestamp))


def drop_partial_edges(
    samples: Sequence[Observation], sampling_step: timedelta
) -> list[Observation]:
    """Retire les mesures du premier et du dernier godet horaire s'ils sont tronques.

    Une extraction par plage commence et finit rarement pile a l'heure : le premier et
    le dernier godet peuvent ne contenir qu'une fraction des mesures attendues au pas
    source, ce qui biaiserait leur moyenne vers les valeurs de bord. Les godets
    interieurs, meme incomplets a cause de valeurs nulles, restent : c'est le role
    d'aggregate_to_hourly d'en faire la moyenne des mesures connues.

    Args:
        samples: Observations d'un seul site, a un pas quelconque.
        sampling_step: Pas source, tel qu'inferred par infer_sampling_step.

    Returns:
        Les memes observations, moins celles des godets de bord tronques.
    """
    if not samples:
        return []

    expected_per_bucket = max(1, round(timedelta(hours=1) / sampling_step))
    ordered = sorted(samples, key=lambda sample: sample.timestamp)
    bucket_of = [_bucket_start(sample.timestamp) for sample in ordered]
    counts = Counter(bucket_of)
    first_bucket, last_bucket = bucket_of[0], bucket_of[-1]

    return [
        sample
        for sample, bucket in zip(ordered, bucket_of, strict=True)
        if not (
            (bucket == first_bucket and counts[first_bucket] < expected_per_bucket)
            or (bucket == last_bucket and counts[last_bucket] < expected_per_bucket)
        )
    ]


def _bucket_start(instant: datetime) -> datetime:
    return instant.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    known = [value for value in values if value is not None]
    if not known:
        return None
    return sum(known) / len(known)
