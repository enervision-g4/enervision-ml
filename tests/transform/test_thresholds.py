import pytest

from enervision_ml.transform.thresholds import compute_threshold_kw


def test_the_threshold_is_a_fraction_of_the_installed_capacity() -> None:
    threshold = compute_threshold_kw(capacity_kw=200, observed_consumption_kw=[], ratio=0.85)

    assert threshold == pytest.approx(170.0)


def test_an_unknown_capacity_falls_back_to_the_observed_high_percentile() -> None:
    observed = [float(value) for value in range(1, 101)]  # 1..100

    threshold = compute_threshold_kw(capacity_kw=None, observed_consumption_kw=observed)

    # Le 95e centile d'une serie 1..100 se situe pres de 95-96.
    assert 94.0 < threshold < 97.0  # type: ignore[operator]


def test_an_unknown_capacity_without_any_history_returns_no_threshold() -> None:
    threshold = compute_threshold_kw(capacity_kw=None, observed_consumption_kw=[])

    assert threshold is None


def test_a_known_capacity_ignores_the_observed_history() -> None:
    threshold = compute_threshold_kw(capacity_kw=100, observed_consumption_kw=[1000.0] * 10)

    assert threshold == pytest.approx(85.0)
