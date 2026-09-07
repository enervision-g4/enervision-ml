from typer.testing import CliRunner

from enervision_ml.cli import application

runner = CliRunner()


def test_the_help_screen_is_reachable_without_any_configuration() -> None:
    result = runner.invoke(application, ["--help"])

    assert result.exit_code == 0
