from enervision_ml.transform.model_version import build_model_version


def test_the_same_feature_contract_yields_the_same_version_across_runs() -> None:
    first_run = build_model_version("scikit-learn==1.9.0")
    second_run = build_model_version("scikit-learn==1.9.0")

    assert first_run == second_run


def test_the_version_does_not_depend_on_the_training_data() -> None:
    # build_model_version ne recoit aucune donnee d'entrainement : deux "runs" sur des
    # jeux de donnees differents mais avec le meme contrat et la meme bibliotheque
    # produisent forcement la meme empreinte.
    training_run_one = build_model_version("scikit-learn==1.9.0")
    training_run_two = build_model_version("scikit-learn==1.9.0")

    assert training_run_one == training_run_two


def test_a_different_estimator_version_yields_a_different_model_version() -> None:
    version_a = build_model_version("scikit-learn==1.9.0")
    version_b = build_model_version("scikit-learn==1.8.0")

    assert version_a != version_b


def test_the_version_names_the_estimator_library() -> None:
    version = build_model_version("scikit-learn==1.9.0")

    assert version.startswith("scikit-learn==1.9.0")
