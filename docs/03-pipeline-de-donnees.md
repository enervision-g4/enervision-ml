# 3. Le pipeline de données

## 3.1 Les cinq couches

Le dépôt suit la même séparation que `enervision-etl` : chaque couche a un rôle unique, et
les dépendances vont toujours dans le même sens.

```mermaid
flowchart TB
    subgraph EX["extract/ - lire"]
        E1["csv_history.py<br/>fichier local"]
        E2["database_history.py<br/>measure_imputed"]
        E3["site_catalog.py<br/>référentiel + capacity_kw"]
    end

    subgraph TR["transform/ - calculer (fonctions pures)"]
        T1["aggregation<br/>features"]
        T2["baseline<br/>evaluation"]
        T3["thresholds<br/>recommendations"]
        T4["weather_outlook<br/>prediction_drafts<br/>model_version"]
    end

    subgraph MO["model/ - apprendre"]
        M1["estimator.py<br/>scikit-learn"]
        M2["forecaster.py<br/>1 modèle = 1 site"]
    end

    subgraph LO["load/ - écrire"]
        L1["prediction_repository"]
        L2["recommendation_repository"]
    end

    subgraph OR["orchestration/ - assembler"]
        O1["EvaluationRun"]
        O2["ForecastRun"]
        O3["DriftFreeScheduler<br/>ShutdownRequest"]
        O4["ExperimentTrackingLogger"]
    end

    EX --> OR
    TR --> OR
    MO --> OR
    OR --> LO
    OR -.-> MLF["MLflow"]

    style OR fill:#e8f0fe,stroke:#4285f4
    style TR fill:#e6f4ea,stroke:#34a853
```

Règle structurante : **`transform/` ne contient que des fonctions pures** (mêmes entrées,
mêmes sorties, aucun effet de bord, aucun SQL, aucune horloge). C'est ce qui rend les 146
tests unitaires du dépôt exécutables sans base de données, sans réseau et sans MLflow.

Inversement, **`orchestration/` ne contient aucune règle métier ni aucun SQL** : il
n'assemble que les couches situées en dessous.

## 3.2 Les deux sources d'historique, derrière un seul contrat

```mermaid
classDiagram
    class HistorySourceLike {
        <<Protocol>>
        +load_observations(site_id) Sequence~Observation~
        +load_site_catalog() Sequence~SiteReference~
    }
    class CsvHistorySource {
        csv_path
        source_timezone
    }
    class DatabaseHistorySource {
        connection
        lookback = 30 jours
    }
    HistorySourceLike <|.. CsvHistorySource
    HistorySourceLike <|.. DatabaseHistorySource
```

Un `Protocol` Python plutôt qu'une classe abstraite : les deux implémentations n'héritent
de rien, et un double de test satisfait le contrat sans rien importer de concret.

| | `CsvHistorySource` | `DatabaseHistorySource` |
|---|---|---|
| Source | Fichier du formateur | Table `measure_imputed` |
| Unité d'origine | kWh au pas source | kW déjà normalisés par l'ETL |
| Agrégation horaire | En Python (`transform/aggregation.py`) | En SQL (`date_trunc` + `avg`) |
| `capacity_kw` | Inconnue (repli percentile) | Lue dans la table `site` |
| Utilité | `evaluate` hors infrastructure | Le service réel |

Deux détails qui ont demandé une décision :

- **`date_trunc` plutôt que `time_bucket`** côté SQL. `time_bucket` est la fonction
  TimescaleDB ; `date_trunc` est du PostgreSQL standard. La requête reste exécutable sur un
  Postgres nu, sans dépendre d'une extension qui pourrait ne pas être activée sur
  l'environnement de destination.
- **`WHERE site_id = %s` en clause dédiée**, pas en paramètre nullable
  (`site_id = %s OR %s IS NULL`) : cette dernière forme empêche Postgres d'utiliser l'index
  sur `site_id` dès que le filtre est actif.

## 3.3 De la mesure brute à l'observation horaire

Le CSV du formateur est au pas de 15 minutes et en kWh ; le modèle travaille à l'heure et
en kW. Trois étapes, toutes dans `transform/aggregation.py`.

```mermaid
flowchart TB
    A["Lignes CSV brutes<br/>pas de 15 min, en kWh"] --> B["infer_sampling_step<br/>médiane des écarts"]
    B --> C["Conversion kWh vers kW<br/>facteur = 1 h / pas"]
    C --> D["drop_partial_edges<br/>retire les godets de bord tronqués"]
    D --> E["aggregate_to_hourly<br/>MOYENNE par heure UTC"]
    E --> F["Observation horaire<br/>site, heure, kW, temp, humidité"]

    style F fill:#e8f0fe,stroke:#4285f4
```

**Pourquoi la médiane des écarts** et non le premier écart observé : un seul intervalle
irrégulier en tête de série (retard de collecte, resynchronisation) fausserait toute la
conversion kWh vers kW.

**Pourquoi une moyenne et jamais une somme** : c'est le piège classique de ce pipeline.

```mermaid
flowchart LR
    subgraph bug["Somme (faux)"]
        direction TB
        S1["heure complète<br/>4 mesures x 100 = 400"]
        S2["heure où l'ETL a redémarré<br/>2 mesures x 100 = 200"]
        S3["Le modèle apprend un CREUX<br/>qui n'a jamais existé"]
        S1 --> S3
        S2 --> S3
    end
    subgraph ok["Moyenne (juste)"]
        direction TB
        M1["heure complète<br/>moyenne = 100 kW"]
        M2["heure incomplète<br/>moyenne = 100 kW"]
        M3["Une puissance reste<br/>une puissance"]
        M1 --> M3
        M2 --> M3
    end
    style bug fill:#fce8e6,stroke:#ea4335
    style ok fill:#e6f4ea,stroke:#34a853
```

**Pourquoi retirer les godets de bord** : une extraction par plage commence et finit
rarement pile à l'heure. Le premier et le dernier godet horaire peuvent ne contenir qu'une
fraction des mesures attendues, ce qui biaiserait leur moyenne. Les godets intérieurs
incomplets, eux, restent : c'est le rôle de la moyenne d'en tenir compte.

**Un godet sans aucune valeur connue reste absent**, jamais ramené à zéro. Même principe
que l'ETL : on ne détruit jamais l'information « il n'y avait rien ici ».

## 3.4 Un lot de prévision, de bout en bout

```mermaid
sequenceDiagram
    autonumber
    participant S as DriftFreeScheduler
    participant R as ForecastRun
    participant DB as PostgreSQL
    participant M as ConsumptionForecaster
    participant ML as MLflow

    S->>R: tick (toutes les 3600 s)
    R->>DB: SELECT site_id, capacity_kw FROM site

    loop pour chaque site du parc
        R->>DB: historique horaire (30 derniers jours)
        DB-->>R: N observations

        Note over R,ML: 1. On note la copie précédente
        R->>DB: dernière prévision écrite pour l'heure<br/>désormais mesurée
        R->>ML: forecast_mae / forecast_mape

        alt moins de MIN_TRAINING_HOURS (168 h)
            R->>R: site ignoré, journalisé, boucle continue
        else historique suffisant
            Note over R,M: 2. Entraînement
            R->>M: fit(observations)
            Note over R,M: 3. Météo future = climatologie
            R->>R: build_climatology + project_weather
            Note over R,M: 4. Prédiction H+1 à H+24
            R->>M: predict(24 observations futures)
            M-->>R: 24 puissances
            Note over R,DB: 5. Seuil, prévisions, recommandations
            R->>R: compute_threshold_kw
            R->>R: build_prediction_rows (uuid5)
            R->>R: build_recommendations (dépassements)
            R->>DB: INSERT prediction ... ON CONFLICT DO NOTHING
            R->>DB: INSERT recommendation ... ON CONFLICT DO NOTHING
            R->>DB: COMMIT
        end
    end

    R-->>S: ForecastReport (forecast / skipped / failed)
```

L'étape 1 est le **contrôle qualité en production** : avant d'entraîner le nouveau modèle,
le service confronte la prévision qu'il avait faite pour l'heure qui vient enfin d'être
mesurée. Détaillé dans [Évaluation et métriques](04-evaluation-et-metriques.md).

## 3.5 La météo du futur : la climatologie

Prédire la consommation de demain 18 h demande la température de demain 18 h. Aucun
service du projet ne la fournit. La solution retenue est la **moyenne climatologique** du
site.

```mermaid
flowchart LR
    A["Historique du site<br/>30 jours"] --> B["Regroupement par<br/>couple (mois, heure UTC)"]
    B --> C["Moyenne de température<br/>et d'humidité par couple"]
    C --> D["(mars, 18h) = 11,4 degrés<br/>(mars, 03h) = 4,1 degrés"]
    E["Cible : demain 18h,<br/>en mars"] --> F["project_weather"]
    D --> F
    F --> G["température = 11,4<br/>humidité = 62"]

    style G fill:#e8f0fe,stroke:#4285f4
```

Ce n'est pas une prévision météo, c'est une **valeur plausible** : « ce que fait la
température à cette heure-là, à cette période de l'année, sur ce site ». Si le couple
(mois, heure) n'a jamais été observé, la valeur reste absente, et le `NaN` natif du modèle
prend le relais plutôt qu'un zéro inventé.

**Conséquence honnête** : la commande `evaluate` mesure la performance avec la température
*réellement observée*, alors que `forecast` se rabat sur cette climatologie. Le gain
affiché par `evaluate` est donc **légèrement optimiste** par rapport à la production.
L'écart est assumé, documenté, et c'est précisément pour le mesurer qu'existe le suivi de
justesse en production (métriques `forecast_mae` dans MLflow).

## 3.6 Le seuil d'alerte et les recommandations

```mermaid
flowchart TB
    A{"capacity_kw connue<br/>dans la table site ?"}
    A -- oui --> B["seuil = capacity_kw x THRESHOLD_RATIO<br/>(0,85 par défaut)"]
    A -- non --> C{"historique disponible ?"}
    C -- oui --> D["seuil = 95e centile<br/>des consommations observées"]
    C -- non --> E["aucun seuil écrit (NULL)"]

    B --> F{"prévision au-dessus du seuil ?"}
    D --> F
    F -- oui --> G["1 ligne dans recommendation<br/>status = open"]
    F -- non --> H["rien"]
    E --> H

    style G fill:#fef3e8,stroke:#f4a142
```

Le repli sur le 95e centile est un **repère observé** plutôt qu'un seuil inventé : à
défaut de connaître la capacité installée, « le site a rarement dépassé cette valeur »
reste une information exploitable. Sans capacité **et** sans historique, aucun seuil n'est
écrit : mieux vaut un `NULL` qu'un chiffre arbitraire que le dashboard afficherait comme
une vérité.

Une recommandation n'est jamais écrite sans sa prévision : les deux partent dans la **même
transaction**, validée par un seul `COMMIT` par site.

## 3.7 L'idempotence : pourquoi rejouer un lot est sans danger

Un service qui tourne en boucle et qui peut être relancé à tout moment finira par rejouer
un lot. La réponse retenue est un identifiant **déterministe**.

```mermaid
flowchart LR
    A["site_id<br/>SITE001"] --> D["uuid5(NAMESPACE_URL, ...)"]
    B["target_timestamp<br/>2026-03-12T18:00Z"] --> D
    C["model_version<br/>scikit-learn==1.7.2+a3f9..."] --> D
    D --> E["prediction_id<br/>toujours le même<br/>pour ce triplet"]
    E --> F["INSERT ... ON CONFLICT<br/>(site_id, target_timestamp,<br/>model_version, timestamp)<br/>DO NOTHING"]

    style E fill:#e8f0fe,stroke:#4285f4
```

`DO NOTHING` et jamais `DO UPDATE` : **la première écriture gagne**. Rejouer un lot après
un échec partiel ne crée aucun doublon et ne modifie rien de ce qui existe déjà.

Une subtilité portée par le schéma (`enervision-devops/db/init/002_create_tables.sql`) :
la colonne `timestamp` (l'instant du lot) fait partie de la contrainte unique, parce
qu'une hypertable TimescaleDB exige que toute contrainte unique inclue sa colonne de
partitionnement. L'effet voulu est que **deux lots successifs conservent chacun leur
prévision** pour la même heure cible : c'est cet historique de prévisions qui permet
ensuite d'en mesurer la justesse.

## 3.8 Isolation des pannes

```mermaid
flowchart TB
    START["Début du lot"] --> LOOP{"site suivant"}
    LOOP --> H["Charger l'historique"]
    H --> CHK{"assez d'historique ?"}
    CHK -- non --> SKIP["sites_skipped<br/>WARNING journalisé"] --> LOOP
    CHK -- oui --> TRY["train + predict + INSERT"]
    TRY --> OK{"succès ?"}
    OK -- oui --> CM["COMMIT<br/>sites_forecast"] --> LOOP
    OK -- non --> RB["ROLLBACK<br/>exception journalisée<br/>sites_failed"] --> LOOP
    LOOP -- "parc épuisé" --> END["ForecastReport"]

    style CM fill:#e6f4ea,stroke:#34a853
    style RB fill:#fce8e6,stroke:#ea4335
    style SKIP fill:#fef7e0,stroke:#f9ab00
```

Un site en échec est annulé, journalisé, compté dans `sites_failed`, et **la boucle
continue sur le reste du parc**. Un seul site avec des données pathologiques ne prive pas
les six autres de leurs prévisions. C'est la même règle que le collecteur temps réel côté
ETL.

Suite : [Évaluation et métriques](04-evaluation-et-metriques.md)
