"""Remplit measure_imputed d'un historique synthetique.

Sert a tester le service ml (commandes evaluate --source database et forecast) sans
attendre que le consumer de persistance de enervision-etl ait tourne. Ecrit aussi les
sites concernes dans SITE s'ils sont absents : measure_imputed porte une cle etrangere
vers site, et il n'y a aucune raison d'exiger que l'ETL ait deja rempli le referentiel
pour tester ce depot isolement.
"""

import argparse
import math
import os
from datetime import UTC, datetime, timedelta
from typing import Optional

import psycopg
from psycopg import Cursor

DEFAULT_CAPACITY_KW = 500
"""Capacite installee arbitraire, suffisante pour un seuil d'alerte plausible."""

INSERT_SITE_IF_ABSENT = """
    INSERT INTO site (site_id, site_type, site_name, capacity_kw, status)
    VALUES (%s, 'office', %s, %s, 'active')
    ON CONFLICT (site_id) DO NOTHING
"""

INSERT_MEASURE_IF_ABSENT = """
    INSERT INTO measure_imputed
        (site_id, "timestamp", consumption_kw, consumption_kwh,
         temperature_celsius, humidity_percent, imputation_method)
    VALUES (%s, %s, %s, %s, %s, %s, 'none')
    ON CONFLICT (site_id, "timestamp") DO NOTHING
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sites", required=True, help="Identifiants de sites, separes par des virgules."
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Nombre de jours d'historique a generer."
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("DATABASE_URL"),
        help="URL Postgres. Par defaut, la variable d'environnement DATABASE_URL.",
    )
    return parser.parse_args()


def synthetic_measurement(hour_of_day: int) -> tuple[float, float, float]:
    angle = (hour_of_day / 24) * 2 * math.pi
    consumption_kw = 50 + 30 * math.sin(angle - math.pi / 2)
    temperature_celsius = 15 + 8 * math.sin(angle)
    humidity_percent = 60.0
    return consumption_kw, temperature_celsius, humidity_percent


def seed_site(cursor: Cursor, site_id: str, days: int) -> int:
    cursor.execute(
        INSERT_SITE_IF_ABSENT, (site_id, f"Site synthetique {site_id}", DEFAULT_CAPACITY_KW)
    )

    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    current = end - timedelta(days=days)
    inserted_count = 0
    while current < end:
        consumption_kw, temperature_celsius, humidity_percent = synthetic_measurement(
            current.hour
        )
        cursor.execute(
            INSERT_MEASURE_IF_ABSENT,
            (site_id, current, consumption_kw, consumption_kw, temperature_celsius,
             humidity_percent),
        )
        inserted_count += 1
        current += timedelta(hours=1)
    return inserted_count


def main(database_url: Optional[str], site_ids: list[str], days: int) -> None:
    if not database_url:
        raise SystemExit("--database-url or DATABASE_URL is required")

    with (
        psycopg.connect(database_url, autocommit=False) as connection,
        connection.cursor() as cursor,
    ):
        for site_id in site_ids:
            inserted_count = seed_site(cursor, site_id, days)
            print(f"{site_id}: {inserted_count} mesures horaires generees")
        connection.commit()


if __name__ == "__main__":
    arguments = parse_arguments()
    main(
        database_url=arguments.database_url,
        site_ids=[site_id.strip() for site_id in arguments.sites.split(",") if site_id.strip()],
        days=arguments.days,
    )
