# 7. FAQ jury

Réponses courtes aux questions attendues. Chacune renvoie au document qui développe.

## Sur le modèle

**Q. Quel modèle avez-vous utilisé ?**
`HistGradientBoostingRegressor` de scikit-learn : du gradient boosting sur arbres de
décision, dans son implémentation optimisée par histogrammes. Un modèle entraîné par site.
→ [2.4](02-le-modele.md)

**Q. Pourquoi celui-là plutôt qu'un réseau de neurones ?**
Nos données sont **tabulaires** (six colonnes) et courtes (720 lignes par site). Sur ce
type de données et à cette échelle, le gradient boosting reste la référence : un LSTM
demanderait beaucoup plus d'historique, du GPU ou de longues minutes de CPU, et une phase
de réglage d'hyperparamètres, pour un gain qui n'existerait probablement pas.

**Q. Pourquoi pas XGBoost ou LightGBM ?**
Performances équivalentes à cette échelle, mais une dépendance externe de plus à
installer, versionner et sécuriser. `HistGradientBoosting` est déjà dans scikit-learn, que
nous avons de toute façon.

**Q. Pourquoi pas SARIMA ou Prophet, ce sont des séries temporelles ?**
Ce sont des modèles de série temporelle pure : ils extrapolent très bien une saisonnalité,
mais intègrent mal des variables exogènes comme la température, qui est justement un de
nos facteurs les plus explicatifs. Et ils demanderaient un paramétrage par site.

**Q. Pourquoi un modèle par site et pas un seul modèle global ?**
Deux raisons. Métier : un entrepôt frigorifique et un atelier d'usinage n'ont pas le même
profil. Technique : un modèle global ferait de `site_id` une variable catégorielle, et un
arbre de décision **lève une erreur** sur une catégorie jamais vue à l'entraînement — le
service casserait au moment où le parc grandit. → [2.3](02-le-modele.md)

**Q. Avez-vous optimisé les hyperparamètres ?**
Non, volontairement. Une recherche d'hyperparamètres exige un troisième jeu (validation)
distinct du jeu de test ; sur 30 jours d'historique, le risque de sur-ajuster le réglage
lui-même dépasse le gain attendu. Les défauts de scikit-learn sont des valeurs de
référence solides sur données tabulaires.

**Q. Comment gérez-vous les valeurs manquantes ?**
Nous ne les remplissons pas : l'estimateur retenu accepte les `NaN` nativement et apprend
lui-même de quel côté de chaque question envoyer une valeur absente. Un zéro à la place
d'un `NaN` apprendrait une corrélation fausse. C'est un des critères qui a fait retenir cet
algorithme.

## Sur la validation

**Q. Comment prouvez-vous que votre modèle sert à quelque chose ?**
On le compare à une **baseline** : le profil horaire moyen, c'est-à-dire ce qu'un tableur
ferait. Sur les sept sites du jeu de données, le modèle bat la baseline, avec un gain moyen
de **+33,6 %** en erreur absolue moyenne. → [4.5](04-evaluation-et-metriques.md)

**Q. Quelles métriques, et pourquoi celles-là ?**
La **MAE** (erreur absolue moyenne, en kW) parce que son unité est celle du métier et
qu'elle décrit l'erreur typique. La **MAPE** (erreur relative, en pourcent) parce qu'elle
est comparable entre sites de tailles différentes. Nous avons écarté le RMSE : il pénalise
les gros écarts, donc il est dominé par quelques pics exceptionnels.

**Q. Comment découpez-vous train et test ?**
Chronologiquement : les 80 % d'heures les plus anciennes pour entraîner, les 20 % les plus
récentes pour tester. Un découpage aléatoire laisserait le modèle s'entraîner sur des
heures **postérieures** à celles qu'il doit deviner : c'est une fuite de données, et le
score serait gonflé. → [4.3](04-evaluation-et-metriques.md)

**Q. Votre MAPE ne devient-elle pas absurde quand la consommation est nulle ?**
C'est traité : les points dont la valeur observée est sous 1 kW sont **exclus du calcul et
comptés séparément** (`excluded_count`), plutôt que d'exploser la moyenne pour un écart
absolu négligeable.

**Q. Vos chiffres de soutenance valent-ils pour la production ?**
Non, et nous le disons. `evaluate` utilise la température **réellement observée**, alors
qu'en production nous ne connaissons pas la température de H+24 : nous nous rabattons sur
la moyenne climatologique du site. Le gain affiché est donc légèrement optimiste. C'est
précisément pour mesurer l'écart que le service journalise en continu un `forecast_mae`
en conditions réelles. → [4.6](04-evaluation-et-metriques.md)

## Sur MLflow

**Q. Comment MLflow est-il déployé ?**
Conteneur `ghcr.io/mlflow/mlflow:v3.16.0` déclaré dans `enervision-devops/compose/
mlflow.yml`, déployé à la main (`workflow_dispatch`, dev ou prod) comme la base et le
broker, puisqu'il n'y a aucune image à construire. Il vit sur le réseau Docker `g4_net`
sous l'alias `g4_mlflow`, avec un backend **SQLite** dans un volume dédié. Le service ML le
joint par `MLFLOW_TRACKING_URI=http://g4_mlflow:5000`. → [5.2](05-mlflow.md)

**Q. Pourquoi SQLite et pas votre TimescaleDB ?**
Seules des métriques ponctuelles y transitent. Rien n'a besoin d'être une hypertable ni
d'être partagé avec un autre service, et SQLite évite de coupler la santé du suivi à celle
de la base métier.

**Q. Que journalisez-vous exactement ?**
Pour `evaluate` : un run parent portant `source` et `test_ratio`, et un run imbriqué par
site portant `site_id`, `model_version`, `model_mae`, `baseline_mae`, `model_mape`,
`baseline_mape`, `improvement_percent`. Pour `forecast` : un run plat par mesure de
justesse, avec `stage=forecast`, `target_timestamp`, `forecast_mae`, `forecast_mape`.
→ [5.4](05-mlflow.md)

**Q. Et les modèles, vous ne les enregistrez pas dans MLflow ?**
Non, et c'est un choix. Nous ne persistons **aucun** artefact de modèle : entraînement et
prédiction ont lieu dans le même run, et le conteneur n'a aucun volume. Un fichier de
modèle serait perdu à chaque redémarrage. Effet de bord favorable : le modèle est toujours
entraîné sur les 30 derniers jours, il ne peut pas devenir périmé silencieusement. Il n'y
a donc rien à versionner côté artefacts, ni de Model Registry à maintenir.
→ [2.6](02-le-modele.md)

**Q. Comment comparez-vous les métriques d'un modèle à un autre ?**
La règle est que deux résultats ne sont comparables que s'ils portent sur **les mêmes
vérités** : c'est pourquoi `model_mae` et `baseline_mae` sont journalisées dans le même
run. En pratique : on modifie le modèle, `model_version` change automatiquement (c'est un
condensat SHA-256 des features et de la version de scikit-learn), on relance `evaluate`
avec le même CSV et le même `test_ratio`, on coche les deux runs dans l'IHM et on lit
`improvement_percent` puis `model_mae` site par site. Un progrès est une amélioration sur
la **majorité** des sites, pas sur un seul. → [5.5](05-mlflow.md)

**Q. Qu'est-ce qui vous garantit qu'on ne compare pas des choses différentes ?**
`source` et `test_ratio` sont journalisés comme paramètres, donc un run sur un autre jeu se
voit tout de suite ; et `model_version` change tout seul dès qu'une feature est ajoutée,
retirée ou réordonnée. On ne peut pas modifier le contrat de features sans que la trace
change.

**Q. Que se passe-t-il si MLflow tombe ?**
Rien de visible côté production : le suivi est **best-effort**. Une panne émet un
avertissement `mlflow_logging_failed` et l'exception est avalée. `evaluate` reste utilisable
hors ligne, sans aucune variable d'environnement. Un serveur de suivi en panne est un
problème d'observabilité, jamais un problème de production.

## Sur l'architecture et l'exploitation

**Q. Comment le service communique-t-il avec le reste du système ?**
Il ne communique avec personne directement : il lit `measure_imputed` et `site`, il écrit
`prediction` et `recommendation`. La base est le point de rendez-vous. S'il tombe, rien
d'autre ne tombe. → [1.2](01-vue-densemble.md)

**Q. Que se passe-t-il si le service redémarre au milieu d'un lot ?**
Rien de grave. Chaque prévision a un identifiant **déterministe** (`uuid5` sur site, heure
cible et version du modèle) et l'insertion est en `ON CONFLICT DO NOTHING` : rejouer un lot
n'écrit jamais deux fois la même ligne. Et le commit est fait **par site**, donc les sites
déjà traités sont acquis. → [3.7](03-pipeline-de-donnees.md)

**Q. Et si un site a des données pathologiques ?**
Il est annulé (`rollback`), journalisé, compté dans `sites_failed`, et la boucle continue
sur le reste du parc. Un site ne prive jamais les autres de leurs prévisions.
→ [3.8](03-pipeline-de-donnees.md)

**Q. Et si un site n'a pas assez d'historique ?**
En dessous de `MIN_TRAINING_HOURS` (168 heures, soit 7 jours), le site est journalisé en
`WARNING` et ignoré. Le modèle lui-même refuse d'être entraîné sous 24 observations
utilisables.

**Q. Comment arrêtez-vous le conteneur proprement ?**
`SIGTERM` ne fait que lever un drapeau, consulté entre deux cycles et pendant l'attente
(découpée en tranches de 0,25 s). Le lot en cours se termine, puis le processus rend la
main. Sans cela, Python interromprait le processus sans exécuter les blocs `finally`, au
milieu d'une transaction. → [6.4](06-deploiement-et-exploitation.md)

**Q. Comment savez-vous quelle version de code tourne en production ?**
L'image est taguée par le SHA du commit, jamais `latest`, et ce même SHA est injecté comme
variable `BUILD_REF` dans l'image, donc visible dans les journaux du conteneur.

**Q. Comment détecteriez-vous une dérive du modèle ?**
Par la métrique `forecast_mae` journalisée en continu dans MLflow. Si elle monte semaine
après semaine **sans que `model_version` ait changé**, c'est le monde qui a changé, pas le
code : nouveaux équipements, changement d'horaires, saison inhabituelle. Le
réentraînement à chaque lot absorbe déjà une partie de cette dérive.

## La question qui fâche

**Q. Votre modèle n'est-il pas trop simple ?**
Il l'est volontairement, et nous savons exactement ce qui manque : des **features
retardées** (la consommation à H-24 et H-168). C'est le levier de précision le plus
important en prévision de charge, et son absence plafonne le modèle. Nous l'avons écarté en
v1 parce qu'il exige la disponibilité de l'historique récent au moment de prédire et une
stratégie multi-horizon — pour une prévision à H+24, la valeur « 24 heures avant » n'est
elle-même pas encore connue. Le contrat de features est conçu pour l'accueillir sans rien
casser, et `model_version` rendra la comparaison avant/après immédiatement lisible dans
MLflow. C'est la prochaine itération, pas un oubli.

Retour à l'[index](README.md)
