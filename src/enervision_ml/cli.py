"""Interface en ligne de commande du service de prevision.

Composition root unique : c'est ici, et seulement ici, que la configuration est
chargee et que les objets concrets (connexion, estimateur, source d'historique) sont
assembles.
"""

from typing import Optional

import typer

from .config import load_forecast_settings
from .extract.csv_history import CsvHistorySource
from .extract.database_history import DatabaseHistorySource
from .logging_setup import configure_logging, get_logger
from .orchestration.evaluation_run import EvaluationRun
from .orchestration.forecast_run import ForecastRun
from .postgres_connection import create_connection

logger = get_logger("cli")

application = typer.Typer(
    help="Service de prevision EnerVision : entrainement par site et ecriture des previsions.",
    add_completion=False,
    # Une erreur metier doit se lire, pas se decoder dans une trace Python.
    pretty_exceptions_enable=False,
)


@application.callback()
def _root() -> None:
    """Service de prevision EnerVision.

    Sans sous-commande, Typer ne construit aucune racine pour --help : ce
    callback vide lui en donne une, meme avant que forecast et evaluate
    n'existent.
    """


@application.command("evaluate")
def evaluate(
    source: str = typer.Option(
        "csv", "--source", help="Origine de l'historique. Seul csv est supporte pour l'instant."
    ),
    csv_path: Optional[str] = typer.Option(
        None, "--csv-path", help="Chemin du CSV, obligatoire quand --source vaut csv."
    ),
    csv_source_timezone: str = typer.Option(
        "UTC", "--csv-source-timezone", help="Fuseau d'ancrage des horodatages naifs du CSV."
    ),
    test_ratio: float = typer.Option(
        0.2, "--test-ratio", help="Fraction de l'historique la plus recente reservee au test."
    ),
) -> None:
    """Compare le modele a la baseline profil-horaire, sans rien ecrire en base.

    Preuve, sans aucune infrastructure, que le modele apprend plus qu'un moyennage
    naif par heure de la journee.

    Args:
        source: Origine de l'historique. Seul csv est supporte pour l'instant ;
            database arrive a l'etape 6.
        csv_path: Chemin du fichier CSV, obligatoire quand source vaut csv.
        csv_source_timezone: Fuseau d'ancrage des horodatages naifs du CSV.
        test_ratio: Fraction de l'historique la plus recente reservee au test.

    Raises:
        typer.Exit: Si la source demandee n'est pas supportee, si --csv-path manque,
            ou si aucun site n'a assez d'historique pour etre evalue.
    """
    configure_logging()

    if source != "csv":
        logger.error("unsupported_evaluation_source", source=source)
        raise typer.Exit(code=1)
    if not csv_path:
        logger.error("missing_csv_path")
        raise typer.Exit(code=1)

    history_source = CsvHistorySource(csv_path, source_timezone=csv_source_timezone)
    report = EvaluationRun(history_source, test_ratio=test_ratio).run()

    if not report.evaluations:
        logger.error("no_site_evaluated", sites_skipped=report.sites_skipped)
        raise typer.Exit(code=1)

    for site_evaluation in report.evaluations:
        typer.echo(
            f"{site_evaluation.site_id:<10} "
            f"model_mae={site_evaluation.model_mae:7.2f} kW  "
            f"baseline_mae={site_evaluation.baseline_mae:7.2f} kW  "
            f"model_mape={site_evaluation.model_mape.value:6.2f} %  "
            f"baseline_mape={site_evaluation.baseline_mape.value:6.2f} %  "
            f"gain={site_evaluation.improvement_percent:+6.1f} %"
        )
    if report.sites_skipped:
        typer.echo(f"Sites ignores (historique insuffisant) : {', '.join(report.sites_skipped)}")

    average_improvement = sum(
        site_evaluation.improvement_percent for site_evaluation in report.evaluations
    ) / len(report.evaluations)
    typer.echo(f"\nGain moyen du modele sur la baseline : {average_improvement:+.1f} %")


@application.command("forecast")
def forecast(
    once: bool = typer.Option(
        False,
        "--once",
        help=(
            "Execute un seul lot puis s'arrete. Obligatoire pour l'instant : la "
            "boucle continue arrive a l'etape 8."
        ),
    ),
) -> None:
    """Entraine un modele par site sur measure_imputed et ecrit ses previsions en base.

    Args:
        once: Execute un seul lot puis s'arrete.

    Raises:
        typer.Exit: Si --once n'est pas fourni, ou si au moins un site a echoue.
    """
    if not once:
        logger.error("continuous_forecast_not_yet_available")
        raise typer.Exit(code=1)

    settings = load_forecast_settings()
    configure_logging(settings.log_level, settings.log_as_json)

    connection = create_connection(settings.database_url)
    try:
        history_source = DatabaseHistorySource(connection)
        report = ForecastRun(
            history_source=history_source,
            connection=connection,
            horizon_hours=settings.horizon_hours,
            minimum_training_hours=settings.min_training_hours,
            threshold_ratio=settings.threshold_ratio,
        ).run()
    finally:
        connection.close()

    logger.info(
        "forecast_completed",
        sites_forecast=len(report.sites_forecast),
        sites_skipped=len(report.sites_skipped),
        sites_failed=len(report.sites_failed),
    )
    if report.sites_failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    application()
