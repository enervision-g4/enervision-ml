"""Empreinte du contrat de modele, cle d'idempotence des previsions en base.

Deux runs avec le meme contrat de features et la meme bibliotheque d'estimateur
produisent la meme empreinte, quel que soit l'historique d'entrainement : elle
identifie un contrat, pas une experimentation particuliere.
"""

import hashlib

from .features import FEATURE_NAMES


def build_model_version(estimator_library_version: str) -> str:
    """Construit une empreinte du contrat de features et de la bibliotheque du modele.

    Args:
        estimator_library_version: Version de la bibliotheque de l'estimateur, par
            exemple "scikit-learn==1.9.0".

    Returns:
        Une chaine commencant par estimator_library_version, suivie d'une empreinte
        courte et stable du contrat de features.
    """
    payload = "|".join((*FEATURE_NAMES, estimator_library_version))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{estimator_library_version}+{digest[:12]}"
