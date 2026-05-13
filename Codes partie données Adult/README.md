# Régression Logistique : Codes données Adult
**Auteurs :** Nassim Khaliss et Yacine Bettani
**Année universitaire :** 2025-2026

## Présentation du Projet
Ce dossier contient l'implémentation accompagnant le partie étude du jeu de données Adult de la deuxième partie du Chapitre 4. La chaîne de traitement est hybride :

1. **R** : Nettoyage des données, gestion de la colinéarité, inférence statistique asymptotique (Fisher, Wald, LRT) et évaluation (Validation Croisée, ROC).

2. **C++ (implémentation native)** : Moteur d'optimisation numérique implémentant la Descente de Gradient, le SGD par Mini-Batch, la méthode BFGS et l'algorithme Newton-Raphson (IRLS) avec régularisation Ridge.

## Architecture du Répertoire
Pour que les flux de données entre R et C++ fonctionnent nativement, **tous les fichiers doivent être placés dans le même dossier racine**. 

```text
📁 Codes partie données Adult/
├── 📄 README.md
├── 📁 Eigen/                     # Bibliothèque mathématique C++ (à inclure)
├── 📄 adult.data                 # Jeu de données
├── 📄 adult.names                # Noms des variables
├── 📄 adult.test                 # Jeu de données test
├── 📄 0_Visualisation_Donnees.R  # Script R (Exploration initiale optionnelle)
├── 📄 1_Preparation_Donnees.R    # Script R (Préparation)
├── 📄 main.cpp                   # Code source C++ (Optimisation)
└── 📄 3_Analyse_Resultats.R      # Script R (Inférence et Graphes)
```

## Prérequis Techniques
* **R (version 4.0+)** avec les packages suivants installés : `ggplot2`, `pROC`, `caret`.
* **Compilateur C++** supportant la norme **C++11** minimum (ex: `g++` ou `clang++`).
* **Bibliothèque Eigen3** (pour le calcul matriciel en C++). *Note : Le dossier Eigen est déjà inclus dans cette archive, aucune installation supplémentaire n'est requise.*

---

## Instructions d'Exécution (Reproductibilité)

L'exécution doit suivre un ordre strict pour respecter le flux de transfert des données entre R et C++. Placez votre terminal à la racine du dossier `Codes partie données Adult/`.

### Étape 1 : Préparation de la matrice de Design (R)
Ce script gère le *One-Hot Encoding*, supprime la colinéarité parfaite, standardise les variables continues et exporte les matrices d'entraînement et de test.

```bash
Rscript 1_Preparation_Donnees.R
```
*Outputs générés dans le dossier courant : `export_adult_train.csv`, `export_adult_test.csv`.*

### Étape 2 : Optimisation Numérique et Estimation (C++)
**Attention :** Pour reproduire les temps de calcul (Benchmarks) mentionnés dans le mémoire (Chapitre 4), il est impératif de compiler avec le flag d'optimisation maximale `-O3`. Il faut également lier la bibliothèque locale Eigen.

1. **Compilation :**
*(Utilisation du flag `-I .` pour lier le dossier local `Eigen`)*
```bash
g++ -std=c++11 -O3 -I . main.cpp -o solver_logistique
```

2. **Exécution :**
```bash
./solver_logistique
```
*Outputs générés dans le dossier courant : `beta_adult_train.csv`.*

### Étape 3 : Inférence Statistique et Évaluation (R)
Ce script récupère les coefficients optimaux ($\hat{\beta}$) calculés par le C++ pour construire la matrice d'Information de Fisher, effectuer les tests de Wald (mise en évidence du paradoxe de Hauck-Donner) et évaluer la capacité de généralisation du modèle.

```bash
Rscript 3_Analyse_Resultats.R
```
*Outputs générés dans le dossier courant : Affichage des métriques console, matrice de confusion, et export des 4 graphiques analytiques (`.png`).*

## Notes aux Évaluateurs
* **Graines aléatoires :** Les générateurs pseudo-aléatoires (`set.seed` en R, `mt19937` en C++) sont fixés pour garantir la stricte reproductibilité des plis de validation croisée et des trajectoires du Mini-Batch SGD.
