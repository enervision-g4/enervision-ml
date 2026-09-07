"""Interface en ligne de commande du service de prevision.

Composition root unique : c'est ici, et seulement ici, que la configuration est
chargee et que les objets concrets (connexion, estimateur, source d'historique) sont
assembles. Les commandes forecast et evaluate sont ajoutees au fil des etapes
suivantes, au-dessus de ce squelette.
"""

import typer

application = typer.Typer(
    help="Service de prevision EnerVision : entrainement par site et ecriture des previsions.",
    add_completion=False,
    # Une erreur metier doit se lire, pas se decoder dans une trace Python.
    pretty_exceptions_enable=False,
)


@application.callback()
def _root() -> None:
    """Service de prevision EnerVision.

    Sans sous-commande, Typer ne construit aucune racine pour --help : ce
    callback vide lui en donne une, meme avant que forecast et evaluate
    n'existent.
    """


if __name__ == "__main__":
    application()
