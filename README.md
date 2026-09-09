# enervision-ml

Service de prevision de consommation du projet EnerVision. Il entraine un modele par
site sur l'historique recent (`measure_imputed`), ecrit 24 heures de prevision dans la
table `prediction`, et calcule un seuil d'alerte a partir de la capacite installee du
site.

## Documentation

Ce README couvre l'installation, la configuration et l'usage. La justification des choix
— quel modèle, pourquoi celui-là, comment il est évalué, comment MLflow est déployé et
comment on compare deux modèles — vit dans [`docs/`](docs/README.md), rédigée pour être
lue sans connaissance préalable en machine learning.

| Document | Contenu |
|---|---|
| [1. Vue d'ensemble](docs/01-vue-densemble.md) | Le problème métier, la place dans l'architecture, le vocabulaire |
| [2. Le modèle](docs/02-le-modele.md) | `HistGradientBoostingRegressor`, les alternatives écartées, `model_version` |
| [3. Le pipeline de données](docs/03-pipeline-de-donnees.md) | Les cinq couches, l'agrégation horaire, l'idempotence |
| [4. Évaluation et métriques](docs/04-evaluation-et-metriques.md) | Baseline, MAE / MAPE, backtest et justesse en production |
| [5. MLflow](docs/05-mlflow.md) | Déploiement, contenu des runs, comparaison de modèles |
| [6. Déploiement et exploitation](docs/06-deploiement-et-exploitation.md) | Image, CI/CD, compose, arrêt propre |
| [7. FAQ jury](docs/07-faq-jury.md) | Les questions attendues et leurs réponses courtes |

## Principes directeurs

- **Un modele par site.** Les identifiants de site different selon la source
  (CSV du formateur ou base), et un arbre de decision refuse une categorie inconnue a
  la prediction : `site_id` n'est donc jamais une feature.
- **Aucun artefact de modele.** Entrainement et prediction ont lieu dans le meme run,
  sans fichier intermediaire : `enervision-devops/compose/ml.yml` ne declare aucun
  volume, un artefact serait perdu a chaque redemarrage du conteneur.
- **Idempotence par construction.** L'identifiant de chaque prevision est deterministe
  (uuid5 sur site, heure visee et version du modele) : rejouer un lot n'ecrit jamais
  deux fois la meme ligne, grace a la contrainte SQL portee par `enervision-devops`
  (`db/migrations/001_add_prediction_unique_constraint.sql`).
- **Un site en echec n'arrete pas le parc.** `forecast` committe apres chaque site ;
  un echec est journalise, annule, et compte dans `sites_failed`, la boucle continue.

## Installation

Le projet cible Python 3.14 et utilise `uv` pour l'environnement et le verrouillage
des dependances.

```bash
git clone https://github.com/enervision-g4/enervision-ml.git
cd enervision-ml

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync
```

## Configuration

Le fichier `.env` n'est pas versionne, a creer a partir du modele :

```bash
cp .env.example .env
```

| Variable | Role | Defaut |
|---|---|---|
| `DATABASE_URL` | Base source (`measure_imputed`) et destination des previsions | *(obligatoire)* |
| `TRAINING_SOURCE` | `database` ou `csv` | `database` |
| `CSV_PATH` | Chemin du CSV, obligatoire si `TRAINING_SOURCE=csv` | — |
| `CSV_SOURCE_TIMEZONE` | Fuseau d'ancrage des horodatages naifs du CSV | `UTC` |
| `FORECAST_INTERVAL_SECONDS` | Cadence de la boucle `forecast` (sans `--once`) | `3600` |
| `HORIZON_HOURS` | Nombre d'heures de prevision ecrites par site et par lot | `24` |
| `MIN_TRAINING_HOURS` | Historique minimal exige avant d'entrainer un site | `168` |
| `THRESHOLD_RATIO` | Fraction de `capacity_kw` retenue comme seuil d'alerte | `0.85` |
| `LOG_LEVEL` / `LOG_AS_JSON` | Journalisation | `INFO` / `true` |

Seule `DATABASE_URL` est obligatoire : `enervision-devops/compose/ml.yml` ne fixe que
cette variable, et reste valide sans etre modifie.

## Utilisation

### `evaluate` — la preuve, sans aucune infrastructure

Compare le modele a une baseline (profil horaire moyen), sur un fichier CSV local :

```bash
uv run enervision-ml evaluate --source csv --csv-path <fichier.csv> --test-ratio 0.2
```

Affiche, par site, l'erreur absolue moyenne (MAE) et l'erreur relative moyenne (MAPE)
du modele et de la baseline, puis le gain moyen. Sur le jeu de donnees fourni pour le
projet, le modele bat la baseline sur les sept sites, avec un gain moyen de +33,6 %.

### `forecast` — le service reel

```bash
DATABASE_URL=postgres://... uv run enervision-ml forecast --once   # un seul lot
DATABASE_URL=postgres://... uv run enervision-ml forecast          # boucle continue
```

Sans `--once`, la commande boucle a la cadence `FORECAST_INTERVAL_SECONDS` jusqu'a un
signal d'arret (`SIGTERM` ou `SIGINT`), termine le lot en cours puis rend la main
proprement. C'est le mode utilise par le conteneur : `restart: unless-stopped` dans
`ml.yml` relancerait un processus qui se termine de lui-meme, voir *Limites connues*.

## Architecture

```
extract/        Lecture : CSV (csv_history.py) ou base (database_history.py),
                 derriere le contrat commun HistorySourceLike.
transform/       Fonctions pures : features, agregation horaire, baseline,
                 evaluation, seuils, previsions, empreinte du modele.
model/           Estimateur scikit-learn (HistGradientBoostingRegressor) et son
                 encapsulation par site, ConsumptionForecaster.
load/            Ecriture idempotente des previsions (ON CONFLICT DO NOTHING).
orchestration/   Assemble les couches ci-dessus : EvaluationRun, ForecastRun,
                 la boucle continue (DriftFreeScheduler) et l'arret propre
                 (ShutdownRequest). Aucune regle metier ni SQL ici.
```

Le paquet `postgres_connection.py` vit a la racine, pas dans `extract/` ni `load/` :
une meme connexion sert la lecture de l'historique et l'ecriture des previsions dans
une seule transaction par site.

## Tests

```bash
uv run pytest                              # suite unitaire, sans infrastructure
uv run ruff check src tests scripts
uv run mypy
```

Les tests d'integration exigent un PostgreSQL reel et sont exclus par defaut :

```bash
docker run -d --name g4_test_db -e POSTGRES_USER=g4_app -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=g4_db -p 5433:5432 \
  -v "$PWD/../enervision-devops/db/init:/docker-entrypoint-initdb.d:ro" \
  timescale/timescaledb:latest-pg16

ENERVISION_TEST_DATABASE_URL=postgres://g4_app:test@localhost:5433/g4_db \
  uv run pytest tests/integration -m integration
```

`scripts/seed_measure_imputed.py` peuple une base locale d'un historique synthetique,
utile pour tester `forecast` sans attendre que le consumer de persistance ait tourne :

```bash
DATABASE_URL=postgres://g4_app:test@localhost:5433/g4_db \
  uv run python scripts/seed_measure_imputed.py --sites SITE001,SITE002 --days 30
```

## Conteneur

```bash
docker build -t enervision-ml .
docker run --rm --env-file .env enervision-ml forecast --once
```

L'image finale pese environ 650 Mo (`scikit-learn`, `scipy`, `numpy`), contre ~80 Mo
pour `enervision-etl` : la chaine scientifique compilee domine la taille, il n'y a rien
de raisonnable a faire pour la reduire davantage sans changer d'estimateur.

## Limites connues

1. **`restart: unless-stopped` et redemarrage.** `ml.yml` relance le conteneur meme
   sur un arret volontaire (code 0). `CMD ["forecast"]` boucle donc en interne plutot
   que de s'appuyer sur un redemarrage Docker par lot. `restart: on-failure` serait
   plus juste ; a signaler a l'equipe devops.
2. **`scripts/restore.sh` efface les previsions cote Azure.** Une restauration
   (`pg_restore --clean --if-exists`) recree les tables depuis le dump on-premise :
   toute prevision produite localement disparait. La migration de contrainte doit etre
   rejouee apres chaque restauration. Point ouvert avec l'equipe devops.
3. **Meteo optimiste dans `evaluate`.** Le MAE affiche utilise la temperature
   observee, alors qu'une prevision H+24 reelle (commande `forecast`) se rabat sur la
   moyenne climatologique (mois, heure) via `transform/weather_outlook.py`, faute de
   connaitre la temperature future. Le gain mesure par `evaluate` est donc legerement
   optimiste par rapport a la production.
4. **Pas de features retardees (lag 24 h, lag 168 h).** C'est le levier de precision
   le plus important en prevision de charge ; son absence plafonne le modele. Ecarte
   sciemment en v1 : il faudrait la disponibilite de l'historique recent au moment de
   predire et une strategie multi-horizon. `FEATURE_NAMES` est concu pour l'accueillir
   plus tard sans rien casser.
