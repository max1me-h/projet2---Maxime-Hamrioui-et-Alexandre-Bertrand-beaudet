# Projet 2 - INF1007 Automne 2026

## Directives

:alarm_clock: Date de remise : selon les consignes de votre cours / enseignant.

:mailbox_with_mail: À remettre selon les modalités indiquées sur Moodle.

## Introduction

Dans ce projet, vous devrez analyser un jeu de données d'employés afin de prédire le risque d'attrition et d'identifier les facteurs qui influencent la décision d'un employé de quitter son entreprise.

L'objectif principal est d'utiliser des techniques de science des données et d'apprentissage automatique pour mieux comprendre les dynamiques de turnover, puis de proposer des recommandations utiles à un gestionnaire RH ou à une équipe de direction.

Le jeu de données utilisé est le célèbre ensemble de données IBM HR Analytics Employee Attrition & Performance, souvent employé dans les travaux d'analyse de données et de modélisation prédictive. Il contient des informations sur l'âge, le niveau d'éducation, les déplacements professionnels, le salaire, le département, la satisfaction au travail, le niveau d'expérience, la présence d'heures supplémentaires, entre autres.

Le but n'est pas seulement de construire un modèle précis, mais aussi de comprendre quels facteurs sont les plus associés à l'attrition et de pouvoir interpréter les résultats.

## Objectifs du projet

- analyser le jeu de données et repérer les tendances importantes ;
- nettoyer et préparer les données pour l'entraînement d'un modèle ;
- construire un modèle de prédiction de l'attrition ;
- évaluer la performance du modèle à l'aide de mesures pertinentes ;
- interpréter les résultats et formuler des recommandations d'action.

## Installation requise

Ce projet nécessite l'utilisation de Python ainsi que plusieurs bibliothèques de data science.

Avant de commencer, assurez-vous que votre environnement conda `INF1007` est activé dans VS Code :

```bash
conda activate INF1007
```

Ensuite, installez les dépendances nécessaires :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Données du projet

Le fichier principal est :

```plaintext
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

Il contient des informations sur un ensemble d'employés, avec des variables de type :

- démographiques : âge, sexe, état civil ;
- organisationnelles : département, rôle, niveau hiérarchique ;
- RH et performance : satisfaction, engagement, évaluation, ancienneté ;
- salariale : salaire, taux horaire, augmentation, primes ;
- travail : distance domicile-travail, déplacements professionnels, heures supplémentaires ;
- cible : `Attrition` (Oui / Non).

Le jeu de données est typiquement utilisé pour répondre à la question suivante :

> Quels employés sont les plus susceptibles de quitter l'entreprise ?

## Informations sur le projet

### Structure du projet

Le projet est organisé de la manière suivante :

```plaintext
2026A-PR02/
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── visualization.py
├── results/
│   ├── reports/
│   └── models/
├── main.py
├── requirements.txt
└── README.md
```

> Note : la structure exacte peut varier selon votre équipe et votre façon de travailler. L'important est de garder un projet bien organisé, lisible et reproductible.

### Détails sur les fichiers

- Le dossier `data/` contient les données brutes utilisées pour l'analyse.
- Le fichier `WA_Fn-UseC_-HR-Employee-Attrition.csv` contient la base d'apprentissage pour l'analyse et la modélisation.
- Le dossier `notebooks/` peut contenir les analyses exploratoires, le nettoyage, la préparation des données et l'évaluation des modèles.
- Le dossier `src/` contient les fonctions Python réutilisables pour le prétraitement, la visualisation et la modélisation.
- Le dossier `results/` contient les visualisations, rapports et modèles exportés.
- Le fichier `main.py` sert en général à exécuter le pipeline complet du projet.

### Variables d'intérêt principales

Parmi les variables du jeu de données, certaines sont particulièrement importantes pour l'analyse de l'attrition :

- `Age` : âge de l'employé ;
- `BusinessTravel` : fréquence des déplacements professionnels ;
- `Department` : département d'appartenance ;
- `DistanceFromHome` : distance domicile-travail ;
- `Education` : niveau d'éducation ;
- `Gender` : sexe ;
- `JobRole` : poste occupé ;
- `MonthlyIncome` : salaire mensuel ;
- `OverTime` : heures supplémentaires ;
- `JobSatisfaction` : satisfaction au travail ;
- `EnvironmentSatisfaction` : satisfaction de l'environnement de travail ;
- `YearsAtCompany` : ancienneté ;
- `YearsInCurrentRole`, `YearsSinceLastPromotion` : expérience dans le poste et promotions ;
- `Attrition` : variable cible.

## Travail à réaliser

Vous devez réaliser une analyse complète du jeu de données et produire un projet structuré et bien documenté.

### Partie 1 : Analyse exploratoire des données

Dans cette première étape, vous devrez :

- charger les données et vérifier leur structure ;
- identifier les colonnes importantes ;
- détecter les valeurs manquantes, les incohérences et les types de variables ;
- analyser la distribution des variables et les corrélations ;
- comparer les employés qui ont quitté l'entreprise à ceux qui sont restés ;
- produire des visualisations claires et interprétables.

### Partie 2 : Prétraitement et préparation des données

Dans cette partie, vous devrez :

- traiter les variables catégorielles ;
- encoder les caractéristiques textuelles ou qualitatives ;
- gérer les variables non numériques s'il y a lieu ;
- normaliser ou standardiser les variables si nécessaire ;
- séparer les données en ensembles d'entraînement et de test ;
- vérifier qu'il n'y a pas de fuite de données ou de biais dans la préparation.

### Partie 3 : Modélisation prédictive

Dans cette étape, vous devrez :

- tester plusieurs modèles de classification, par exemple :
  - régression logistique ;
  - arbre de décision ;
  - forêt aléatoire ;
  - gradient boosting ;
  - SVM ou XGBoost si relevant du cours ;
- choisir un ou plusieurs modèles pertinents ;
- évaluer leur performance à l'aide de métriques comme :
  - précision ;
  - rappel ;
  - F1-score ;
  - courbe ROC-AUC ;
  - matrice de confusion.

### Partie 4 : Interprétation et recommandations

Après la modélisation, il faut :

- identifier les variables les plus influentes ;
- expliquer les résultats de manière claire ;
- formuler des recommandations concrètes pour une équipe RH ;
- conclure sur les facteurs qui semblent le plus fortement liés à l'attrition.

## Exemples de questions à explorer

Voici quelques pistes de recherche pour guider votre analyse :

- Les employés avec beaucoup d'heures supplémentaires ont-ils plus de chances de quitter ?
- La distance domicile-travail est-elle corrélée au risque d'attrition ?
- Les employés avec une faible satisfaction au travail sont-ils plus susceptibles de partir ?
- Quel département présente la plus forte proportion d'attrition ?
- Les promotions ou l'ancienneté jouent-ils un rôle important ?
- Quel poste ou niveau hiérarchique est le plus exposé au risque ?

## Recommandations générales

- Faites des visualisations propres et lisibles.
- Gardez votre code organisé et bien commenté.
- Évitez de travailler uniquement sur l'exactitude ; l'interprétation compte autant que la performance.
- Utilisez les métriques adaptées à un problème de classification déséquilibrée.
- Communiquez vos conclusions avec une logique business et RH.

## Évaluation attendue

Une bonne remise de ce projet devrait inclure :

- une analyse exploratoire rigoureuse ;
- un traitement des données bien justifié ;
- une modélisation claire et cohérente ;
- une comparaison des modèles, si possible ;
- une interprétation des résultats ;
- une conclusion pertinente et des recommandations utiles.

## Barème de correction (exemple)

| Partie | Tâche | Points |
|---|---|---:|
| Analyse exploratoire | Nettoyage, visualisation, compréhension des données | /4 |
| Prétraitement | Encodage, séparation des données, préparation du modèle | /3 |
| Modélisation | Choix et entraînement de modèles | /5 |
| Évaluation | Précision, F1, confusion matrix, interprétation | /4 |
| Analyse métier | Facteurs clés, recommandations, conclusion | /4 |
| **Total** |  | **/20** |

## Conclusion

Le projet 2 d'INF1007 vise à mettre en pratique les notions de science des données et d'apprentissage automatique dans un contexte concret et pertinent : l'analyse du turnover en entreprise. Au-delà de la simple prédiction, l'objectif est de comprendre les causes profondes de l'attrition afin d'aider les organisations à mieux gérer leur capital humain.

Ce type de projet permet de développer à la fois des compétences techniques (prétraitement, modélisation, évaluation) et des compétences analytiques (interprétation, communication, recommandations).

---

# Projet 2 - INF1007
# Analyse de l'attrition des employés

## Résumé

Ce projet consiste à analyser le jeu de données IBM HR Analytics Employee Attrition & Performance afin d'identifier les facteurs associés à l'attrition et à prédire les employés susceptibles de quitter l'organisation.

## But du projet

- prédire l'attrition à l'aide de modèles de machine learning ;
- identifier les variables les plus importantes ;
- interpréter les résultats dans un contexte RH ;
- proposer des recommandations concrètes aux décideurs.

## Fichier de données

```plaintext
WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## Stack technique recommandée

```bash
python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Exemple de pipeline de travail

1. Charger les données.
2. Explorer les variables.
3. Nettoyer et préparer les données.
4. Construire un modèle de classification.
5. Évaluer les performances.
6. Interpréter les résultats.
7. Rédiger les recommandations.

## À retenir

La valeur du projet ne réside pas uniquement dans la précision du modèle, mais aussi dans la qualité de l'analyse, la clarté des visualisations et l'interprétation des facteurs de risque associés à l'attrition.

"""

message":"Met à jour le README du projet 2 avec une structure inspirée du projet 1 et adaptée au sujet d’analyse de l’attrition des employés.","owner":"max1me-h","path":"README.md","repo":"projet2---Maxime-Hamrioui-et-Alexandre-Bertrand-beaudet