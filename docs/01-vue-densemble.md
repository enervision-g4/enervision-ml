# 1. Vue d'ensemble

## 1.1 Le problème métier

EnerVision supervise un parc de sites industriels équipés de compteurs électriques.
Les autres services du projet racontent le **passé** : ce qui a été consommé, ce qui a
déclenché une alerte.

`enervision-ml` est le seul service qui parle du **futur** : *combien ce site va-t-il
consommer dans les 24 prochaines heures, et va-t-il dépasser sa capacité ?*

L'intérêt est opérationnel : savoir à 14 h qu'un site dépassera son seuil à 18 h laisse
quatre heures pour déplacer une charge, alors qu'une alerte à 18 h ne laisse que le
constat.

```mermaid
flowchart LR
    A["<b>PASSÉ</b><br/>measure_raw<br/>measure_imputed<br/>alert"] --> B["<b>PRÉSENT</b><br/>dashboard<br/>temps réel"]
    B --> C["<b>FUTUR</b><br/>prediction<br/>recommendation"]
    A -.- A1["produit par enervision-etl"]
    C -.- C1["produit par enervision-ml"]

    style C fill:#e8f0fe,stroke:#4285f4
```

## 1.2 Place dans l'architecture

Le service ne parle à personne directement : il lit une table, il écrit deux tables. La
base est le point de rendez-vous entre les équipes.

```mermaid
flowchart LR
    subgraph collecte["enervision-etl"]
        API["API Mock<br/>capteurs"] --> COL["Collecteur"]
        COL --> KAFKA[("Kafka")]
        KAFKA --> CONS["Consumer<br/>persistance"]
    end

    CONS --> DB[("PostgreSQL / TimescaleDB")]

    subgraph ml["enervision-ml (ce dépôt)"]
        direction TB
        F["forecast<br/>1 lot / heure"]
    end

    DB -- "lit measure_imputed + site" --> F
    F -- "écrit prediction + recommendation" --> DB
    F -. "métriques" .-> MLF["MLflow<br/>serveur de suivi"]

    DB --> APIS["enervision-api"] --> DASH["enervision-dashboard"]

    style ml fill:#e8f0fe,stroke:#4285f4
    style MLF fill:#fef3e8,stroke:#f4a142
```

Conséquences directes de ce choix :

- **aucun couplage réseau** avec l'ETL ou l'API : si le service ML tombe, rien d'autre ne
  tombe, le dashboard affiche simplement des prévisions qui vieillissent ;
- **une seule connexion PostgreSQL** sert la lecture de l'historique et l'écriture des
  prévisions, dans une transaction par site.

## 1.3 Les deux commandes

Le dépôt expose une CLI à deux commandes, qui répondent à deux besoins différents.

```mermaid
flowchart TB
    CLI["enervision-ml"]
    CLI --> EV["evaluate"]
    CLI --> FC["forecast"]

    EV --> EV1["Lit un CSV ou la base"]
    EV1 --> EV2["Entraîne, compare à une baseline"]
    EV2 --> EV3["Affiche MAE / MAPE / gain<br/>N'ÉCRIT RIEN EN BASE"]

    FC --> FC1["Lit measure_imputed"]
    FC1 --> FC2["Entraîne, prédit H+1 à H+24"]
    FC2 --> FC3["Écrit prediction + recommendation"]

    style EV3 fill:#e6f4ea,stroke:#34a853
    style FC3 fill:#fce8e6,stroke:#ea4335
```

| Commande | Rôle | Écrit en base ? | Besoin d'infrastructure ? |
|---|---|---|---|
| `evaluate` | Prouver que le modèle apprend quelque chose | Non | Non (un CSV suffit) |
| `forecast` | Le service réel, en boucle dans un conteneur | Oui | Oui (PostgreSQL) |

`evaluate` est la commande de la soutenance : elle se lance sur un portable, sans base,
sans Docker, et sort un tableau chiffré.

## 1.4 Le vocabulaire minimal

Cinq mots suffisent pour lire le reste de cette documentation.

```mermaid
flowchart LR
    A["<b>Observation</b><br/>une heure de mesure<br/>d'un site"] --> B["<b>Features</b><br/>ce qu'on donne<br/>au modèle"]
    A --> C["<b>Cible (target)</b><br/>ce qu'on veut<br/>qu'il devine"]
    B --> D["<b>Entraînement</b><br/>le modèle cherche le lien<br/>features - cible"]
    C --> D
    D --> E["<b>Prédiction</b><br/>on donne des features<br/>sans cible, il répond"]
    F["<b>Baseline</b><br/>une méthode bêtement simple,<br/>pour avoir un point de comparaison"] -.-> E
```

- **Observation** : une ligne de données. Ici, *le site SITE001, le 12 mars à 14 h, a
  consommé 320 kW, il faisait 8 degrés et 60 % d'humidité*.
- **Features** (variables d'entrée) : les colonnes qu'on autorise le modèle à regarder.
  Ici : heure, jour, week-end, mois, température, humidité.
- **Cible** : la colonne qu'on lui demande de deviner. Ici : `consumption_kw`.
- **Entraînement** : on montre au modèle quelques milliers d'observations complètes
  (features **et** cible), il en déduit des règles.
- **Baseline** : une méthode volontairement simpliste. Si le modèle ne la bat pas, il ne
  sert à rien, et c'est cette comparaison qui rend le chiffre crédible.

## 1.5 Les quatre principes directeurs du dépôt

Ils expliquent la majorité des choix techniques détaillés dans les documents suivants.

```mermaid
flowchart TB
    P1["<b>Un modèle par site</b><br/>jamais un modèle global"]
    P2["<b>Aucun artefact de modèle</b><br/>entraînement et prédiction<br/>dans le même run"]
    P3["<b>Idempotence par construction</b><br/>rejouer un lot n'écrit<br/>jamais deux fois"]
    P4["<b>Un site en échec<br/>n'arrête pas le parc</b><br/>commit par site"]

    P1 --> R1["site_id n'est jamais une feature"]
    P2 --> R2["aucun volume dans ml.yml,<br/>rien à versionner, rien à perdre"]
    P3 --> R3["prediction_id déterministe (uuid5)<br/>+ ON CONFLICT DO NOTHING"]
    P4 --> R4["try / rollback / log / continue"]
```

Suite : [Le modèle](02-le-modele.md)
