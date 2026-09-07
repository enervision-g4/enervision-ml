"""Configuration du service, lue dans l'environnement et validee au demarrage.

Aucune adresse en dur, et une configuration incomplete fait echouer le demarrage
immediatement plutot qu'apres plusieurs minutes de fonctionnement. Seule DATABASE_URL
est obligatoire : `enervision-devops/compose/ml.yml` ne fixe que cette variable, et
reste valide sans etre modifie tant que le reste garde un defaut.
"""

import re
from typing import Literal, Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ACCEPTED_DATABASE_SCHEMES = ("postgres://", "postgresql://")
"""Schemas acceptes pour l'URL de la base. libpq reconnait les deux."""

SQLALCHEMY_DRIVER_SUFFIX = re.compile(r"^(postgres|postgresql)\+\w+://")
"""SQLAlchemy ecrit parfois postgresql+psycopg:// dans le DATABASE_URL partage entre
tous les services du parc. Ce schema n'a de sens que pour SQLAlchemy ; psycopg, utilise
ici directement, attend postgres:// ou postgresql:// nu."""

TrainingSource = Literal["database", "csv"]


class ForecastSettings(BaseSettings):
    """Parametres du service de prevision.

    Attributes:
        database_url: URL de connexion a PostgreSQL, source de lecture par defaut et
            destination des previsions.
        training_source: Origine de l'historique d'entrainement.
        csv_path: Chemin du fichier CSV, obligatoire quand training_source vaut csv.
        csv_source_timezone: Fuseau des horodatages naifs du CSV.
        forecast_interval_seconds: Cadence de la boucle de la commande forecast.
        horizon_hours: Nombre d'heures de prevision ecrites par site a chaque lot.
        min_training_hours: Historique minimal exige avant d'entrainer un site.
        threshold_ratio: Fraction de capacity_kw au-dela de laquelle une prevision
            declenche une recommandation.
        log_level: Seuil de journalisation.
        log_as_json: Vrai pour une sortie JSON, faux pour un rendu lisible en console.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str
    training_source: TrainingSource = "database"
    csv_path: Optional[str] = None
    csv_source_timezone: str = "UTC"

    forecast_interval_seconds: int = 3600
    horizon_hours: int = 24
    min_training_hours: int = 168
    threshold_ratio: float = 0.85

    log_level: str = "INFO"
    log_as_json: bool = True

    @model_validator(mode="before")
    @classmethod
    def strip_surrounding_whitespace(
        cls,
        submitted_values: dict[str, object],
    ) -> dict[str, object]:
        """Retire les espaces et retours chariot autour de chaque valeur.

        Un fichier .env enregistre sous Windows termine ses lignes par un retour
        chariot, que docker transmet tel quel dans l'environnement du conteneur.

        Args:
            submitted_values: Valeurs brutes issues de l'environnement.

        Returns:
            Les memes valeurs, chaines nettoyees.
        """
        if not isinstance(submitted_values, dict):
            return submitted_values
        return {
            name: value.strip() if isinstance(value, str) else value
            for name, value in submitted_values.items()
        }

    @field_validator("database_url", mode="before")
    @classmethod
    def strip_the_sqlalchemy_driver_suffix(cls, configured_url: object) -> object:
        """Retire un suffixe de pilote SQLAlchemy (dialect+driver://) du schema.

        Args:
            configured_url: Valeur brute lue dans l'environnement.

        Returns:
            L'URL avec le suffixe de pilote retire ; inchangee si absente ou si la
            valeur n'est pas une chaine.
        """
        if isinstance(configured_url, str):
            return SQLALCHEMY_DRIVER_SUFFIX.sub(r"\1://", configured_url)
        return configured_url

    @field_validator("database_url")
    @classmethod
    def require_a_known_database_scheme(cls, configured_url: str) -> str:
        """Verifie que l'URL de la base porte un schema reconnu.

        Args:
            configured_url: Valeur brute lue dans l'environnement.

        Returns:
            L'URL inchangee.

        Raises:
            ValueError: Si l'URL ne commence par aucun schema accepte.
        """
        if not configured_url.startswith(ACCEPTED_DATABASE_SCHEMES):
            raise ValueError(
                f"DATABASE_URL must start with one of {ACCEPTED_DATABASE_SCHEMES}, "
                f"received {configured_url!r}"
            )
        return configured_url

    @model_validator(mode="after")
    def require_a_csv_path_when_the_source_is_csv(self) -> "ForecastSettings":
        """Verifie qu'un chemin est fourni quand la source d'entrainement est csv.

        Returns:
            Les parametres inchanges.

        Raises:
            ValueError: Si training_source vaut csv sans csv_path renseigne.
        """
        if self.training_source == "csv" and not self.csv_path:
            raise ValueError("CSV_PATH is required when TRAINING_SOURCE is csv")
        return self


def load_forecast_settings() -> ForecastSettings:
    """Charge la configuration du service.

    Returns:
        Les parametres valides du service.

    Raises:
        ValidationError: Si une variable obligatoire manque ou est invalide.
    """
    return ForecastSettings()  # type: ignore[call-arg]
