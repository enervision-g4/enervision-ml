# Documentation `enervision-ml`

Le service de prévision de consommation du projet EnerVision : ce qu'il fait, comment il
le fait, et pourquoi il a été construit ainsi.

Cette documentation est écrite pour être lue **sans connaissance préalable en machine
learning**. Chaque notion est introduite au moment où elle sert.

## Parcours de lecture

| # | Document | Ce qu'on y trouve |
|---|---|---|
| 1 | [Vue d'ensemble](01-vue-densemble.md) | Le problème métier, la place du service dans l'architecture, le vocabulaire minimal |
| 2 | [Le modèle](02-le-modele.md) | Ce que le modèle apprend, le choix de l'algorithme et les alternatives écartées |
| 3 | [Le pipeline de données](03-pipeline-de-donnees.md) | Du CSV / de la base jusqu'aux lignes écrites, couche par couche |
| 4 | [Évaluation et métriques](04-evaluation-et-metriques.md) | Comment on prouve que le modèle sert à quelque chose |
| 5 | [MLflow](05-mlflow.md) | Le suivi d'expériences : déploiement, contenu des runs, comparaison de modèles |
| 6 | [Déploiement et exploitation](06-deploiement-et-exploitation.md) | Image, CI/CD, compose, limites de ressources |
| 7 | [FAQ jury](07-faq-jury.md) | Les questions attendues et leurs réponses courtes |

## Résumé en dix lignes

Le service lit l'historique de consommation horaire de chaque site, entraîne **un modèle
par site** sur cet historique, et écrit **24 heures de prévision** dans la table
`prediction`. Quand une prévision dépasse le seuil d'alerte du site, il écrit en plus une
ligne dans `recommendation`.

Le modèle est un **HistGradientBoostingRegressor** (scikit-learn). Il apprend la relation
entre six variables d'entrée (heure, jour de la semaine, week-end, mois, température,
humidité) et la puissance appelée.

La preuve de son utilité est une comparaison chiffrée contre une **baseline** naïve : le
profil horaire moyen. Sur le jeu de données du projet, le modèle bat cette baseline sur
les sept sites, avec un gain moyen de **+33,6 %** en erreur absolue moyenne.

Les métriques sont journalisées dans un serveur **MLflow** déployé à côté des services,
qui permet de comparer deux versions de modèle entre elles.
