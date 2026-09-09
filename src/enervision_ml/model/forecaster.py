"""Modele de prevision pour un seul site : etat mutable assume.

Contrairement au reste du paquet, qui ne manipule que des fonctions pures et des
enregistrements figes, un estimateur est un objet a etat : il est entraine une fois
puis interroge plusieurs fois. C'est pourquoi ce module vit dans son propre
sous-paquet plutot qu'au milieu de transform/.
"""

from collections.abc import Sequence
from typing import Optional

from ..records import Observation
from ..transform.features import build_feature_row
from ..transform.model_version import build_model_version
from .errors import NotEnoughSamplesError, UnfittedModelError
from .estimator import EstimatorLike, create_estimator, estimator_library_version

MINIMUM_TRAINING_SAMPLES = 24
"""En dessous, meme une seule journee complete d'historique n'est pas couverte."""


class ConsumptionForecaster:
    """Modele de prevision de consommation pour un seul site.

    Attributes:
        site_id: Site pour lequel ce modele est entraine.
        model_version: Empreinte du contrat de features et de l'estimateur, connue
            des la construction, independamment de l'entrainement.
        estimator_name: Nom de la classe de l'estimateur utilise (ex.
            "HistGradientBoostingRegressor"), pour l'identifier lisiblement dans
            MLflow sans decoder l'empreinte model_version.
    """

    def __init__(self, site_id: str, estimator: Optional[EstimatorLike] = None) -> None:
        """Prepare un modele non entraine.

        Args:
            site_id: Site pour lequel ce modele sera entraine.
            estimator: Estimateur a utiliser, injecte par les tests. Un
                HistGradientBoostingRegressor reel par defaut.
        """
        self.site_id = site_id
        self._estimator = estimator if estimator is not None else create_estimator()
        self.model_version = build_model_version(estimator_library_version())
        self.estimator_name = type(self._estimator).__name__
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Vrai une fois fit() applique avec succes."""
        return self._is_fitted

    def fit(self, observations: Sequence[Observation]) -> None:
        """Entraine le modele sur les observations dont la consommation est connue.

        Args:
            observations: Historique d'entrainement du site.

        Raises:
            NotEnoughSamplesError: Si trop peu d'observations ont une consommation
                connue pour entrainer utilement l'estimateur.
        """
        usable = [
            observation for observation in observations if observation.consumption_kw is not None
        ]
        if len(usable) < MINIMUM_TRAINING_SAMPLES:
            raise NotEnoughSamplesError(self.site_id, len(usable), MINIMUM_TRAINING_SAMPLES)

        features_matrix: list[tuple[float, ...]] = [
            build_feature_row(observation) for observation in usable
        ]
        target: list[float] = [
            observation.consumption_kw
            for observation in usable
            if observation.consumption_kw is not None
        ]
        self._estimator.fit(features_matrix, target)
        self._is_fitted = True

    def predict(self, observations: Sequence[Observation]) -> list[float]:
        """Predit une puissance pour chaque observation fournie.

        Args:
            observations: Observations dont deriver les features de prediction. Leur
                consumption_kw n'est pas lu : seuls le calendaire et la meteo le sont.

        Returns:
            Une puissance predite par observation, dans le meme ordre.

        Raises:
            UnfittedModelError: Si le modele n'a pas ete entraine au prealable.
        """
        if not self._is_fitted:
            raise UnfittedModelError(self.site_id)

        features_matrix: list[tuple[float, ...]] = [
            build_feature_row(observation) for observation in observations
        ]
        return [float(value) for value in self._estimator.predict(features_matrix)]
