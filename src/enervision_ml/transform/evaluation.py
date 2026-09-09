"""Comparaison chiffree du modele et de la baseline.

Sert exclusivement a la commande evaluate : rien ici n'ecrit en base, ces fonctions ne
font que mesurer un ecart entre des valeurs deja predites et deja connues.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from ..records import Observation

NEAR_ZERO_TRUTH_KW = 1.0
"""En dessous de cette puissance, une erreur relative explose pour un ecart negligeable."""


@dataclass(frozen=True)
class MapeResult:
    """Erreur relative moyenne, avec le compte des verites proches de zero exclues.

    Attributes:
        value: Erreur relative moyenne, en pourcentage, sur les verites exploitables.
        excluded_count: Nombre de points exclus car la verite etait proche de zero.
    """

    value: float
    excluded_count: int


@dataclass(frozen=True)
class SiteEvaluation:
    """Comparaison du modele et de la baseline pour un site.

    Attributes:
        site_id: Site evalue.
        model_version: Empreinte du contrat de features et de l'estimateur ayant
            produit cette evaluation.
        estimator_name: Nom de la classe de l'estimateur utilise (ex.
            "HistGradientBoostingRegressor"), lisible directement dans MLflow.
        model_mae: Erreur absolue moyenne du modele.
        baseline_mae: Erreur absolue moyenne de la baseline (profil horaire).
        model_mape: Erreur relative moyenne du modele.
        baseline_mape: Erreur relative moyenne de la baseline.
        improvement_percent: Gain du modele sur la baseline en MAE, negatif si le
            modele est moins bon.
    """

    site_id: str
    model_version: str
    estimator_name: str
    model_mae: float
    baseline_mae: float
    model_mape: MapeResult
    baseline_mape: MapeResult
    improvement_percent: float


def split_chronologically(
    observations: Sequence[Observation], test_ratio: float
) -> tuple[list[Observation], list[Observation]]:
    """Separe l'historique en entrainement et test, dans l'ordre chronologique.

    Une separation aleatoire laisserait le modele s'entrainer sur des heures
    posterieures a celles qu'il doit predire au moment du test, ce qui surestimerait
    sa performance reelle.

    Args:
        observations: Historique complet du site, dans un ordre quelconque.
        test_ratio: Fraction des observations les plus recentes reservee au test,
            strictement entre 0 et 1.

    Returns:
        Le couple (entrainement, test), chacun trie par instant croissant.

    Raises:
        ValueError: Si test_ratio n'est pas strictement entre 0 et 1.
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio must be strictly between 0 and 1, received {test_ratio}")

    ordered = sorted(observations, key=lambda observation: observation.timestamp)
    split_index = round(len(ordered) * (1 - test_ratio))
    return ordered[:split_index], ordered[split_index:]


def mean_absolute_error(truths: Sequence[float], predictions: Sequence[float]) -> float:
    """Calcule l'erreur absolue moyenne.

    Args:
        truths: Valeurs observees.
        predictions: Valeurs predites, dans le meme ordre.

    Returns:
        La moyenne des ecarts absolus.

    Raises:
        ValueError: Si les deux sequences n'ont pas la meme longueur, ou sont vides.
    """
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length")
    if not truths:
        raise ValueError("cannot compute a mean absolute error over an empty sequence")
    return sum(
        abs(truth - prediction) for truth, prediction in zip(truths, predictions, strict=True)
    ) / len(truths)


def mean_absolute_percentage_error(
    truths: Sequence[float], predictions: Sequence[float]
) -> MapeResult:
    """Calcule l'erreur relative moyenne, en excluant les verites proches de zero.

    Args:
        truths: Valeurs observees.
        predictions: Valeurs predites, dans le meme ordre.

    Returns:
        L'erreur relative moyenne sur les points exploitables, et le compte des
        points exclus.

    Raises:
        ValueError: Si les deux sequences n'ont pas la meme longueur, ou si tous les
            points sont exclus.
    """
    if len(truths) != len(predictions):
        raise ValueError("truths and predictions must have the same length")

    usable = [
        (truth, prediction)
        for truth, prediction in zip(truths, predictions, strict=True)
        if abs(truth) >= NEAR_ZERO_TRUTH_KW
    ]
    excluded_count = len(truths) - len(usable)
    if not usable:
        raise ValueError("every truth was excluded as near zero: cannot compute a percentage error")

    value = 100 * sum(abs((truth - prediction) / truth) for truth, prediction in usable) / len(
        usable
    )
    return MapeResult(value=value, excluded_count=excluded_count)


def evaluate_site(
    site_id: str,
    truths: Sequence[float],
    model_predictions: Sequence[float],
    baseline_predictions: Sequence[float],
    model_version: str,
    estimator_name: str,
) -> SiteEvaluation:
    """Compare le modele et la baseline sur les memes verites.

    Args:
        site_id: Site evalue.
        truths: Valeurs observees sur la periode de test.
        model_predictions: Predictions du modele, dans le meme ordre.
        baseline_predictions: Predictions de la baseline, dans le meme ordre.
        model_version: Empreinte du contrat de features et de l'estimateur utilise.
        estimator_name: Nom de la classe de l'estimateur utilise.

    Returns:
        La comparaison chiffree des deux, voir SiteEvaluation.
    """
    model_mae = mean_absolute_error(truths, model_predictions)
    baseline_mae = mean_absolute_error(truths, baseline_predictions)
    improvement_percent = (
        100 * (baseline_mae - model_mae) / baseline_mae if baseline_mae else 0.0
    )
    return SiteEvaluation(
        site_id=site_id,
        model_version=model_version,
        estimator_name=estimator_name,
        model_mae=model_mae,
        baseline_mae=baseline_mae,
        model_mape=mean_absolute_percentage_error(truths, model_predictions),
        baseline_mape=mean_absolute_percentage_error(truths, baseline_predictions),
        improvement_percent=improvement_percent,
    )
