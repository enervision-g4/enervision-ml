import csv
import math
from datetime import datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from enervision_ml.cli import application

runner = CliRunner()

HEADER = [
    "timestamp", "site_id", "site_type", "site_name", "consumption_kwh", "consumption_euros",
    "temperature_celsius", "humidity_percent", "solar_irradiance_wm2", "hour", "day_of_week",
    "day_name", "month", "is_weekend", "is_working_hours",
]


def write_synthetic_history(csv_path: Path, days: int = 10) -> None:
    # Un cycle journalier sinusoidal, plus un effet week-end que seule une baseline
    # dotee de la feature is_weekend peut capturer : la profil horaire moyen (qui ne
    # connait que l'heure) melange les deux, le modele les distingue.
    start = datetime(2024, 1, 1)  # un lundi
    rows = []
    for hour_index in range(days * 24):
        timestamp = start + timedelta(hours=hour_index)
        hour_effect = 50 + 30 * math.sin((timestamp.hour / 24) * 2 * math.pi - math.pi / 2)
        weekend_effect = 20 if timestamp.weekday() >= 5 else 0
        consumption = hour_effect + weekend_effect
        rows.append(
            [
                timestamp.strftime("%Y-%m-%d %H:%M:%S"), "SITE001", "office", "Bureau",
                f"{consumption:.2f}", "0", "18.0", "60", "0", str(timestamp.hour),
                str(timestamp.weekday()), timestamp.strftime("%A"), str(timestamp.month), "0", "0",
            ]
        )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def test_the_help_screen_is_reachable_without_any_configuration() -> None:
    result = runner.invoke(application, ["--help"])

    assert result.exit_code == 0


def test_evaluate_reports_a_positive_gain_on_a_learnable_pattern(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    write_synthetic_history(csv_path)

    result = runner.invoke(
        application,
        ["evaluate", "--source", "csv", "--csv-path", str(csv_path), "--test-ratio", "0.2"],
    )

    assert result.exit_code == 0, result.output
    assert "SITE001" in result.output
    assert "Gain moyen" in result.output


def test_evaluate_without_a_csv_path_fails_clearly() -> None:
    result = runner.invoke(application, ["evaluate", "--source", "csv"])

    assert result.exit_code != 0


def test_evaluate_rejects_an_unsupported_source() -> None:
    result = runner.invoke(application, ["evaluate", "--source", "database"])

    assert result.exit_code != 0
