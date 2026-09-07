"""Caracterise un CSV d'historique avant de figer sa lecture.

A lancer sur le vrai fichier avant d'ecrire csv_history.py : le pas d'echantillonnage
fixe la conversion kWh -> kW, et les site_id doivent etre confrontes a ceux de la base
avant qu'un modele ne soit entraine sur un referentiel qui ne correspond a rien en
production. Ce script ne depend que de la bibliotheque standard : il doit rester
utilisable avant meme que le lecteur officiel n'existe.
"""

import csv
import sys
from collections import Counter
from datetime import datetime
from itertools import pairwise
from pathlib import Path

REQUIRED_COLUMNS = (
    "timestamp",
    "site_id",
    "consumption_kwh",
    "temperature_celsius",
    "humidity_percent",
)

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def section_title(texte: str) -> None:
    print(f"\n{'=' * 72}\n{texte}\n{'=' * 72}")


def parse_timestamp(raw_value: str) -> datetime:
    for candidate_format in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw_value, candidate_format)
        except ValueError:
            continue
    raise ValueError(f"unrecognised timestamp format: {raw_value!r}")


if len(sys.argv) != 2:
    print("usage: probe_training_csv.py <chemin-du-csv>")
    sys.exit(1)

csv_path = Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
print(f"Fichier : {csv_path}")
print(f"Lignes  : {len(rows)}")

section_title("1. Colonnes")
present_columns = set(rows[0].keys()) if rows else set()
missing_columns = [name for name in REQUIRED_COLUMNS if name not in present_columns]
print(f"  Colonnes presentes : {sorted(present_columns)}")
if missing_columns:
    print(f"  Colonnes obligatoires absentes : {missing_columns}")
else:
    print("  Toutes les colonnes obligatoires sont presentes.")

section_title("2. Sites")
site_counts = Counter(row["site_id"] for row in rows)
for site_id, count in sorted(site_counts.items()):
    print(f"  {site_id:<10} {count} lignes")

section_title("3. Horodatages : format, fuseau, pas d'echantillonnage")
sample_raw = rows[0]["timestamp"]
print(f"  Exemple brut : {sample_raw!r}")
has_explicit_offset = ("+" in sample_raw) or (sample_raw.rstrip().endswith("Z"))
print(f"  Marque de fuseau explicite dans le texte : {has_explicit_offset}")

first_site_id = rows[0]["site_id"]
timestamps = sorted(
    parse_timestamp(row["timestamp"]) for row in rows if row["site_id"] == first_site_id
)
gaps_minutes = sorted(
    int((later - earlier).total_seconds() / 60) for earlier, later in pairwise(timestamps)
)
gap_counts = Counter(gaps_minutes)
print(f"  Pas observes (minutes) pour {rows[0]['site_id']} : {gap_counts.most_common(5)}")
median_gap_minutes = gaps_minutes[len(gaps_minutes) // 2] if gaps_minutes else None
print(f"  Pas median retenu : {median_gap_minutes} min")
if median_gap_minutes:
    conversion_factor = 60 / median_gap_minutes
    print(f"  Conversion kWh -> kW retenue : kw = kwh * {conversion_factor:g}")

section_title("4. Valeurs manquantes par colonne")
for column_name in present_columns:
    empty_count = sum(1 for row in rows if row[column_name] == "")
    if empty_count:
        share = 100 * empty_count / len(rows)
        print(f"  {column_name:<24} {empty_count} vides ({share:.1f} %)")

section_title("VERDICT")
if missing_columns:
    print("  Colonnes obligatoires manquantes : ajuster REQUIRED_COLUMNS ou le CSV.")
    sys.exit(1)
print("  Pret pour csv_history.py avec la conversion et le fuseau ci-dessus.")
