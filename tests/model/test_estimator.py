import numpy as np

from enervision_ml.model.estimator import (
    create_estimator,
    estimator_class_name,
    estimator_library_version,
)


def test_estimator_library_version_names_scikit_learn() -> None:
    assert estimator_library_version().startswith("scikit-learn==")


def test_create_estimator_returns_a_new_instance_each_call() -> None:
    assert create_estimator() is not create_estimator()


def test_estimator_class_name_matches_what_create_estimator_builds() -> None:
    assert estimator_class_name() == type(create_estimator()).__name__


def test_the_real_estimator_learns_a_daily_pattern_better_than_its_mean() -> None:
    # Consommation sinusoidale sur 30 jours, un pic a midi et un creux a minuit : un
    # arbre de decision doit apprendre ce cycle bien mieux qu'une moyenne constante.
    hours_since_start = np.arange(24 * 30)
    hour_of_day = (hours_since_start % 24).reshape(-1, 1).astype(float)
    consumption = 50.0 + 30.0 * np.sin((hour_of_day.ravel() / 24.0) * 2 * np.pi - np.pi / 2)

    training_size = 20 * 24  # 20 des 30 jours
    train_features, test_features = hour_of_day[:training_size], hour_of_day[training_size:]
    train_target, test_target = consumption[:training_size], consumption[training_size:]

    estimator = create_estimator()
    estimator.fit(train_features, train_target)
    predictions = estimator.predict(test_features)

    model_mae = float(np.mean(np.abs(test_target - predictions)))
    mean_baseline_mae = float(np.mean(np.abs(test_target - train_target.mean())))

    assert model_mae < mean_baseline_mae
