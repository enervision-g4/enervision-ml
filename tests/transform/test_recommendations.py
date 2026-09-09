from datetime import UTC, datetime
from uuid import uuid4

from enervision_ml.records import PredictionRow
from enervision_ml.transform.recommendations import build_recommendations

MODEL_VERSION = "scikit-learn==1.9.0+abcdef123456"
GENERATED_AT = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_prediction(
    predicted_consumption_kw: float,
    threshold_kw,
    target_timestamp: datetime = datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
) -> PredictionRow:
    return PredictionRow(
        prediction_id=uuid4(),
        site_id="SITE001",
        target_timestamp=target_timestamp,
        predicted_consumption_kw=predicted_consumption_kw,
        threshold_kw=threshold_kw,
        model_version=MODEL_VERSION,
        timestamp=GENERATED_AT,
    )


def test_a_forecast_above_the_threshold_produces_a_recommendation() -> None:
    prediction = make_prediction(predicted_consumption_kw=200.0, threshold_kw=170.0)

    recommendations = build_recommendations([prediction])

    assert len(recommendations) == 1
    assert recommendations[0].prediction_id == prediction.prediction_id
    assert recommendations[0].site_id == "SITE001"


def test_a_forecast_below_the_threshold_produces_no_recommendation() -> None:
    prediction = make_prediction(predicted_consumption_kw=100.0, threshold_kw=170.0)

    assert build_recommendations([prediction]) == []


def test_a_forecast_exactly_at_the_threshold_produces_no_recommendation() -> None:
    # Depasser strictement le seuil, pas seulement l'atteindre : sinon un site qui
    # tourne pile a son seuil declencherait une recommandation en continu.
    prediction = make_prediction(predicted_consumption_kw=170.0, threshold_kw=170.0)

    assert build_recommendations([prediction]) == []


def test_a_forecast_with_no_known_threshold_produces_no_recommendation() -> None:
    prediction = make_prediction(predicted_consumption_kw=1000.0, threshold_kw=None)

    assert build_recommendations([prediction]) == []


def test_the_action_names_the_hour_and_the_two_values() -> None:
    prediction = make_prediction(
        predicted_consumption_kw=200.0,
        threshold_kw=170.0,
        target_timestamp=datetime(2024, 3, 5, 18, 0, tzinfo=UTC),
    )

    recommendation = build_recommendations([prediction])[0]

    assert "2024-03-05 18:00" in recommendation.action_description
    assert "200.0" in recommendation.action_description
    assert "170.0" in recommendation.action_description


def test_only_the_predictions_above_threshold_produce_a_recommendation() -> None:
    above = make_prediction(predicted_consumption_kw=200.0, threshold_kw=170.0)
    below = make_prediction(predicted_consumption_kw=100.0, threshold_kw=170.0)

    recommendations = build_recommendations([above, below])

    assert len(recommendations) == 1
    assert recommendations[0].prediction_id == above.prediction_id
