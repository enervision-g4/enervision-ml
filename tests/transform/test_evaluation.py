from datetime import UTC, datetime, timedelta

import pytest

from enervision_ml.records import Observation
from enervision_ml.transform.evaluation import (
    evaluate_site,
    mean_absolute_error,
    mean_absolute_percentage_error,
    split_chronologically,
)


def make_observations(count: int, shuffled: bool = False) -> list[Observation]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    observations = [
        Observation(
            site_id="SITE001",
            timestamp=start + timedelta(hours=index),
            consumption_kw=float(index),
            temperature_celsius=18.0,
            humidity_percent=60.0,
        )
        for index in range(count)
    ]
    if shuffled:
        # Ordre delibere different de l'ordre chronologique : le split doit trier
        # lui-meme, il ne peut pas se fier a l'ordre d'entree.
        observations = observations[::2] + observations[1::2]
    return observations


def test_the_split_never_puts_a_later_hour_in_the_training_set() -> None:
    observations = make_observations(20, shuffled=True)

    train, test = split_chronologically(observations, test_ratio=0.2)

    assert max(o.timestamp for o in train) < min(o.timestamp for o in test)


def test_the_split_respects_the_requested_ratio() -> None:
    observations = make_observations(10)

    train, test = split_chronologically(observations, test_ratio=0.2)

    assert len(train) == 8
    assert len(test) == 2


def test_a_ratio_outside_the_open_interval_is_refused() -> None:
    observations = make_observations(10)

    for invalid_ratio in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            split_chronologically(observations, test_ratio=invalid_ratio)


def test_mean_absolute_error_averages_the_absolute_gaps() -> None:
    error = mean_absolute_error([10.0, 20.0], [12.0, 15.0])

    assert error == pytest.approx(3.5)


def test_a_near_zero_truth_is_excluded_from_the_percentage_error_and_counted() -> None:
    # Une consommation quasi nulle rendrait l'erreur relative arbitrairement grande
    # pour un ecart absolu negligeable.
    result = mean_absolute_percentage_error([0.05, 100.0], [1.0, 90.0])

    assert result.excluded_count == 1
    assert result.value == pytest.approx(10.0)


def test_the_improvement_is_negative_when_the_model_loses_against_the_baseline() -> None:
    evaluation = evaluate_site(
        site_id="SITE001",
        truths=[10.0, 20.0, 30.0],
        model_predictions=[15.0, 25.0, 35.0],  # ecart constant de 5
        baseline_predictions=[11.0, 21.0, 31.0],  # ecart constant de 1, bien meilleur
    )

    assert evaluation.improvement_percent < 0


def test_the_improvement_is_positive_when_the_model_beats_the_baseline() -> None:
    evaluation = evaluate_site(
        site_id="SITE001",
        truths=[10.0, 20.0, 30.0],
        model_predictions=[11.0, 21.0, 31.0],
        baseline_predictions=[15.0, 25.0, 35.0],
    )

    assert evaluation.improvement_percent > 0
