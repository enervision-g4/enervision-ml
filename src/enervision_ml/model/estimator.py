"""Acces a l'estimateur scikit-learn, importe au dernier moment.

L'import de scikit-learn est differe a l'interieur de create_estimator : le reste du
paquet, y compris forecaster.py, reste important sans la dependance lourde, et les
tests qui injectent un estimateur factice n'ont jamais besoin de l'installer.
"""

from collections.abc import Sequence
from typing import Protocol, cast


class EstimatorLike(Protocol):
    """Sous-ensemble de l'API scikit-learn reellement utilise ici."""

    def fit(self, features_matrix: Sequence[Sequence[float]], target: Sequence[float]) -> object:
        """Ajuste l'estimateur sur une matrice de features et une cible."""
        ...

    def predict(self, features_matrix: Sequence[Sequence[float]]) -> Sequence[float]:
        """Predit une valeur pour chaque ligne d'une matrice de features."""
        ...


def create_estimator(random_state: int = 0) -> EstimatorLike:
    """Cree un nouvel estimateur, non entraine.

    Args:
        random_state: Graine de reproductibilite.

    Returns:
        Un HistGradientBoostingRegressor : NaN natifs, pas de mise a l'echelle a
        maintenir, entrainement rapide sur l'historique d'un site.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    # scikit-learn ne livre pas de py.typed : sans ce cast, mypy voit un retour Any.
    return cast(EstimatorLike, HistGradientBoostingRegressor(random_state=random_state))


def estimator_library_version() -> str:
    """Renvoie la version de scikit-learn utilisee, pour l'empreinte du modele.

    Returns:
        Une chaine de la forme "scikit-learn==X.Y.Z".
    """
    import sklearn

    return f"scikit-learn=={sklearn.__version__}"
