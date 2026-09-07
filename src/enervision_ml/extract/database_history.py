"""Lecture de l'historique horaire depuis measure_imputed.

L'agregation se fait cote SQL, avec date_trunc plutot que la fonction time_bucket de
TimescaleDB : la requete reste executable sur un PostgreSQL nu, sans dependre d'une
extension qui pourrait ne pas etre activee sur l'environnement de destination.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Optional

from ..postgres_connection import ConnectionLike
from ..records import Observation, SiteReference
from .errors import DatabaseQueryError
from .site_catalog import load_sites

SELECT_HOURLY_OBSERVATIONS_FOR_SITE = """
    SELECT date_trunc('hour', "timestamp") AS bucket_start,
           avg(consumption_kw)      AS consumption_kw,
           avg(temperature_celsius) AS temperature_celsius,
           avg(humidity_percent)    AS humidity_percent,
           count(consumption_kw)    AS measured_points
      FROM measure_imputed
     WHERE site_id = %s AND "timestamp" >= %s AND "timestamp" < %s
     GROUP BY bucket_start
    HAVING count(consumption_kw) > 0
     ORDER BY bucket_start
"""
"""avg() et non sum() : une heure ou l'ETL a redemarre n'a que quelques mesures, la
moyenne donne toujours une puissance correcte, la somme donnerait un creux fantome que
le modele apprendrait. Le filtre site_id = %s est une clause dediee plutot qu'un
parametre nullable (site_id = %s OR %s IS NULL) : cette derniere forme empecherait
Postgres d'utiliser l'index sur site_id des que le filtre est actif."""

DEFAULT_LOOKBACK = timedelta(days=30)
"""Fenetre d'entrainement par defaut : assez large pour un modele par site."""


class DatabaseHistorySource:
    """Source d'historique lisant measure_imputed et le referentiel des sites.

    Attributes:
        lookback: Duree d'historique lue avant l'instant courant.
    """

    def __init__(
        self, connection: ConnectionLike, lookback: timedelta = DEFAULT_LOOKBACK
    ) -> None:
        """Prepare la source.

        Args:
            connection: Connexion ouverte vers la base.
            lookback: Duree d'historique a lire avant l'instant courant.
        """
        self._connection = connection
        self.lookback = lookback

    def load_observations(self, site_id: str) -> Sequence[Observation]:
        """Charge l'historique horaire d'un site sur la fenetre lookback.

        Args:
            site_id: Site dont l'historique est demande.

        Returns:
            Les observations horaires du site, triees par instant croissant.

        Raises:
            DatabaseQueryError: Si le pilote echoue a executer la requete.
        """
        end = datetime.now(UTC)
        start = end - self.lookback
        try:
            with self._connection.cursor() as cursor:
                cursor.execute(SELECT_HOURLY_OBSERVATIONS_FOR_SITE, (site_id, start, end))
                rows = cursor.fetchall()
        except Exception as failure:
            raise DatabaseQueryError(site_id, str(failure)) from failure

        return [self._row_to_observation(site_id, row) for row in rows]

    def load_site_catalog(self) -> Sequence[SiteReference]:
        """Charge le referentiel des sites connus de la base.

        Returns:
            Un SiteReference par site, voir site_catalog.load_sites.

        Raises:
            DatabaseQueryError: Si le pilote echoue a executer la requete.
        """
        return load_sites(self._connection)

    @staticmethod
    def _row_to_observation(site_id: str, row: tuple[object, ...]) -> Observation:
        bucket_start, consumption_kw, temperature_celsius, humidity_percent, _measured_points = row
        assert isinstance(bucket_start, datetime)
        return Observation(
            site_id=site_id,
            timestamp=bucket_start.astimezone(UTC),
            consumption_kw=_as_optional_float(consumption_kw),
            temperature_celsius=_as_optional_float(temperature_celsius),
            humidity_percent=_as_optional_float(humidity_percent),
        )


def _as_optional_float(value: object) -> Optional[float]:
    return None if value is None else float(value)  # type: ignore[arg-type]
