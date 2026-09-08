# 4. Évaluation et métriques

## 4.1 La question à laquelle il faut répondre

Un modèle produit toujours des chiffres. La seule question qui compte est : **ces chiffres
valent-ils mieux qu'une méthode que n'importe qui aurait pu écrire en dix minutes ?**

Sans point de comparaison, « MAE = 42 kW » ne veut rien dire : c'est excellent sur un site
qui consomme 2 000 kW, c'est catastrophique sur un site qui en consomme 60.

D'où la démarche du dépôt :

```mermaid
flowchart LR
    A["Historique du site"] --> B["Découpage<br/>chronologique"]
    B --> C["Modèle<br/>HistGradientBoosting"]
    B --> D["Baseline<br/>profil horaire moyen"]
    C --> E["Mêmes heures de test,<br/>mêmes vérités"]
    D --> E
    E --> F["MAE, MAPE, gain en pourcent"]

    style D fill:#fef7e0,stroke:#f9ab00
    style F fill:#e8f0fe,stroke:#4285f4
```

## 4.2 La baseline : le profil horaire moyen

C'est l'adversaire, volontairement simple : pour chaque heure de la journée, la moyenne de
ce que le site a consommé à cette heure-là dans le passé.

```mermaid
flowchart TB
    A["Historique d'entraînement"] --> B["Regrouper par heure UTC"]
    B --> C["00h : moyenne = 95 kW<br/>01h : moyenne = 92 kW<br/>...<br/>14h : moyenne = 410 kW"]
    C --> D["Prévoir demain 14h ?<br/>Réponse : 410 kW"]

    style D fill:#fef7e0,stroke:#f9ab00
```

Elle n'est pas naïve par hasard : c'est exactement ce qu'un tableur ferait. Elle capte le
rythme journalier — le signal le plus fort — mais **ignore tout le reste** : le jour de la
semaine, la saison, la température. Battre cette baseline, c'est prouver que le modèle a
appris quelque chose que la moyenne ne contient pas.

Une précaution de code : si une heure de la journée n'a jamais été observée à
l'entraînement, la baseline retombe sur la moyenne globale plutôt que de perdre le point.
L'évaluation sanctionne alors une baseline incomplète au lieu d'écarter la comparaison.

La baseline n'est **jamais écrite en base** : elle n'existe que pour la commande
`evaluate`.

## 4.3 Le découpage chronologique

Le découpage classique en machine learning est aléatoire. Ici, il serait **faux**.

```mermaid
flowchart TB
    subgraph faux["Découpage aléatoire - INTERDIT ici"]
        direction LR
        F1["lun 8h<br/>TRAIN"] --- F2["lun 9h<br/>TEST"] --- F3["lun 10h<br/>TRAIN"] --- F4["lun 11h<br/>TEST"]
        F5["Le modèle s'entraîne sur des heures<br/>POSTÉRIEURES à celles qu'il doit deviner<br/>= fuite de données, score gonflé"]
    end

    subgraph juste["Découpage chronologique - retenu"]
        direction LR
        J1["jours 1 à 24<br/>TRAIN (80 %)"] --> J2["jours 25 à 30<br/>TEST (20 %)"]
        J3["Exactement la situation réelle :<br/>on connaît le passé,<br/>on devine la suite"]
    end

    style faux fill:#fce8e6,stroke:#ea4335
    style juste fill:#e6f4ea,stroke:#34a853
```

`split_chronologically` trie les observations par instant et coupe : les 80 % les plus
anciennes pour l'entraînement, les 20 % les plus récentes pour le test. Le ratio est
réglable par `--test-ratio`.

## 4.4 Les deux métriques

### MAE — erreur absolue moyenne

```
MAE = moyenne( | valeur observée - valeur prévue | )
```

En kW. Se lit directement : « en moyenne, on se trompe de 42 kW ». C'est la métrique de
référence du service, parce que son unité est celle du métier.

Le choix de la valeur absolue plutôt que du carré (RMSE) est délibéré : le RMSE pénalise
davantage les grosses erreurs, ce qui le rend très sensible aux quelques pics
exceptionnels d'un historique industriel. La MAE décrit l'erreur typique, celle qui
intéresse l'exploitant.

### MAPE — erreur relative moyenne

```
MAPE = 100 x moyenne( | (observé - prévu) / observé | )
```

En pourcentage. Sans unité, donc **comparable entre deux sites** de tailles différentes.

Un piège connu de cette métrique est traité explicitement : quand la valeur observée est
proche de zéro, l'erreur relative explose pour un écart absolu insignifiant (se tromper de
1 kW sur 0,5 kW observé donne 200 %). Les points dont la vérité est sous
`NEAR_ZERO_TRUTH_KW = 1.0` kW sont donc **exclus du calcul et comptés séparément**
(`excluded_count`), plutôt que d'être silencieusement moyennés.

```mermaid
flowchart LR
    A["Points de test"] --> B{"vérité au moins 1 kW<br/>en valeur absolue ?"}
    B -- oui --> C["compte dans la MAPE"]
    B -- non --> D["exclu, mais COMPTÉ<br/>dans excluded_count"]
    C --> E["MapeResult(value, excluded_count)"]
    D --> E

    style E fill:#e8f0fe,stroke:#4285f4
```

### Le gain

```
gain = 100 x (MAE_baseline - MAE_modèle) / MAE_baseline
```

Positif : le modèle fait mieux. Négatif : il fait pire, et il faut le dire.

## 4.5 Le résultat mesuré

```bash
uv run enervision-ml evaluate --source csv --csv-path <fichier.csv> --test-ratio 0.2
```

Sortie, une ligne par site :

```
SITE001    model_mae=  31.42 kW  baseline_mae=  48.77 kW  model_mape=  8.12 %  baseline_mape= 12.90 %  gain= +35.6 %
...
Gain moyen du modele sur la baseline : +33.6 %
```

Sur le jeu de données fourni pour le projet, **le modèle bat la baseline sur les sept
sites**, avec un gain moyen de **+33,6 %** en erreur absolue moyenne.

Aucune infrastructure n'est nécessaire : pas de base, pas de Docker, pas de MLflow. C'est
la commande à lancer en soutenance.

## 4.6 Deux régimes de mesure, à ne pas confondre

Le dépôt mesure la qualité à deux endroits, qui ne racontent pas la même chose.

```mermaid
flowchart TB
    subgraph A["evaluate - BACKTEST"]
        direction TB
        A1["Sur un historique figé"]
        A2["Découpage train/test artificiel"]
        A3["Utilise la température RÉELLE"]
        A4["Compare modèle contre baseline"]
        A5["Métriques : model_mae, baseline_mae,<br/>model_mape, baseline_mape,<br/>improvement_percent"]
        A1 --> A2 --> A3 --> A4 --> A5
    end

    subgraph B["forecast - JUSTESSE EN PRODUCTION"]
        direction TB
        B1["Sur le service qui tourne"]
        B2["Pas de découpage : la vraie prévision"]
        B3["Utilise la climatologie<br/>(comme en production)"]
        B4["Compare prévision passée<br/>contre mesure réelle"]
        B5["Métriques : forecast_mae, forecast_mape"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    A -.->|"optimiste"| C["Écart attendu"]
    B -.->|"la vérité terrain"| C

    style A fill:#e8f0fe,stroke:#4285f4
    style B fill:#e6f4ea,stroke:#34a853
```

Le second régime est celui que décrit `transform/forecast_accuracy.py`, appelé au début de
chaque lot :

```mermaid
sequenceDiagram
    participant T1 as Lot de 14h00
    participant DB as table prediction
    participant T2 as Lot de 15h00
    participant ML as MLflow

    T1->>DB: prévision pour 15h00 = 402 kW
    Note over T1,T2: une heure passe, l'ETL écrit la mesure réelle
    T2->>DB: quelle était la dernière prévision<br/>écrite pour 15h00 ?
    DB-->>T2: 402 kW (model_version X)
    T2->>T2: mesure réelle de 15h00 = 388 kW
    T2->>T2: forecast_mae = |388 - 402| = 14 kW
    T2->>ML: log_forecast_accuracy(stage=forecast)
```

C'est le seul signal qui porte sur le modèle **tel qu'il tourne réellement**, avec le
délai d'une heure nécessaire pour que la vérité soit connue. C'est aussi le signal qui
détecterait une dérive : si `forecast_mae` monte semaine après semaine sans que
`model_version` ait changé, le monde a changé, pas le code.

Ce contrôle est **best-effort** : ni une lecture en échec ni une panne MLflow ne font
échouer le lot de prévision. Une lecture en échec annule proprement la transaction en
cours pour que l'écriture des nouvelles prévisions reparte d'une transaction saine.

## 4.7 Ce que le découpage ne couvre pas

Un seul découpage train/test, même chronologique, reste un unique point de mesure. Un
protocole plus rigoureux serait une **validation croisée glissante** (plusieurs découpages
successifs, chacun testant sur la fenêtre suivante). Elle n'a pas été mise en place à ce
stade : elle multiplierait le temps d'évaluation par le nombre de replis sur un historique
de 30 jours déjà court, et le suivi de justesse en production (4.6) fournit déjà une
mesure continue et non truquée.

Suite : [MLflow](05-mlflow.md)
