"""Lecture de l'historique depuis un CSV, seule source disponible avant tout deploiement.

Le pas d'echantillonnage fixe la conversion kWh -> kW : il est inferred par site plutot
que suppose, une extraction pouvant couvrir des sites a des frequences differentes. Les
colonnes calendaires precalculees (hour, day_of_week, month, ...) sont lues par
DictReader comme le reste de la ligne mais jamais exploitees : elles sont
probablement en heure locale, alors que la reconstruction du calendaire se fait depuis
le seul horodatage, une fois ancre en UTC.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from ..records import Observation, SiteReference
from ..transform.aggregation import aggregate_to_hourly, drop_partial_edges, infer_sampling_step
from .errors import MalformedRowError, MissingColumnError

REQUIRED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "site_id",
    "consumption_kwh",
    "temperature_celsius",
    "humidity_percent",
)
"""Colonnes sans lesquelles aucune observation n'est constructible."""

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_SAMPLING_STEP = timedelta(hours=1)
"""Repli quand un site n'a qu'une seule mesure : rien a inferer, aucune conversion."""


@dataclass(frozen=True)
class _RawSample:
    """Une ligne du CSV, au pas source, avant conversion kWh -> kW."""

    site_id: str
    timestamp: datetime
    consumption_kwh: Optional[float]
    temperature_celsius: Optional[float]
    humidity_percent: Optional[float]


class CsvHistorySource:
    """Source d'historique lisant un CSV au format du generateur du formateur.

    Attributes:
        csv_path: Chemin du fichier lu.
    """

    def __init__(self, csv_path: str, source_timezone: str = "UTC") -> None:
        """Prepare la source, sans lire le fichier avant le premier appel.

        Args:
            csv_path: Chemin du fichier CSV.
            source_timezone: Fuseau d'ancrage des horodatages naifs du fichier.
        """
        self.csv_path = csv_path
        self._zone = ZoneInfo(source_timezone)
        self._rows: Optional[list[dict[str, str]]] = None

    def load_observations(self, site_id: str) -> Sequence[Observation]:
        """Charge l'historique horaire d'un site.

        Args:
            site_id: Site dont l'historique est demande.

        Returns:
            Les observations horaires du site, triees par instant croissant. Vide si
            le site n'apparait pas dans le fichier.

        Raises:
            MissingColumnError: Si une colonne obligatoire est absente de l'entete.
            MalformedRowError: Si une ligne du site ne peut pas etre interpretee.
        """
        matching_rows = self._read_rows_for(site_id)
        if not matching_rows:
            return []

        raw_samples = [self._parse_row(row, row_number) for row_number, row in matching_rows]
        timestamps = [sample.timestamp for sample in raw_samples]
        sampling_step = (
            infer_sampling_step(timestamps) if len(timestamps) >= 2 else DEFAULT_SAMPLING_STEP
        )
        conversion_factor = DEFAULT_SAMPLING_STEP / sampling_step

        converted = [
            Observation(
                site_id=sample.site_id,
                timestamp=sample.timestamp,
                consumption_kw=(
                    None
                    if sample.consumption_kwh is None
                    else sample.consumption_kwh * conversion_factor
                ),
                temperature_celsius=sample.temperature_celsius,
                humidity_percent=sample.humidity_percent,
            )
            for sample in raw_samples
        ]
        trimmed = drop_partial_edges(converted, sampling_step)
        return aggregate_to_hourly(trimmed)

    def load_site_catalog(self) -> Sequence[SiteReference]:
        """Charge le referentiel des sites presents dans le fichier.

        La capacite installee n'est pas portee par ce CSV : elle reste inconnue, le
        seuil d'alerte se rabattra alors sur le percentile observe.

        Returns:
            Un SiteReference par site distinct rencontre, dans l'ordre d'apparition.
        """
        self._ensure_loaded()
        assert self._rows is not None
        distinct_site_ids = dict.fromkeys(row["site_id"] for row in self._rows)
        return [SiteReference(site_id=site_id, capacity_kw=None) for site_id in distinct_site_ids]

    def _read_rows_for(self, site_id: str) -> list[tuple[int, dict[str, str]]]:
        self._ensure_loaded()
        assert self._rows is not None
        return [
            (row_number, row)
            for row_number, row in enumerate(self._rows, start=2)  # l'entete est la ligne 1
            if row["site_id"] == site_id
        ]

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        with Path(self.csv_path).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            for column_name in REQUIRED_COLUMNS:
                if column_name not in header:
                    raise MissingColumnError(column_name, self.csv_path)
            self._rows = list(reader)

    def _parse_row(self, row: dict[str, str], row_number: int) -> _RawSample:
        try:
            naive_timestamp = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
        except ValueError as failure:
            raise MalformedRowError(row_number, f"unparseable timestamp: {failure}") from failure

        anchored = naive_timestamp.replace(tzinfo=self._zone)
        return _RawSample(
            site_id=row["site_id"],
            timestamp=anchored.astimezone(UTC),
            consumption_kwh=_parse_optional_float(row["consumption_kwh"]),
            temperature_celsius=_parse_optional_float(row["temperature_celsius"]),
            humidity_percent=_parse_optional_float(row["humidity_percent"]),
        )


def _parse_optional_float(raw_value: str) -> Optional[float]:
    stripped_value = raw_value.strip()
    return None if not stripped_value else float(stripped_value)
