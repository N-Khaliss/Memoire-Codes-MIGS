# Régression Logistique : Codes données Manning
**Auteurs :** Nassim Khaliss et Yacine Bettani

**Année universitaire :** 2025-2026

## Présentation du Projet
Ce dossier contient l'implémentation accompagnant la partie étude du jeu de données Manning (première partie du Chapitre 4). La chaîne de traitement est hybride :

1. R : Nettoyage des données, sélection du meilleur modèle par minimisation du critère BIC, inférence statistique asymptotique (Fisher, Wald, LRT) et évaluation de la performance (Validation Croisée K-Fold, ROC).
2. C++ (implémentation native) : Moteur d'optimisation numérique implémentant la Descente de Gradient, la méthode BFGS et l'algorithme Newton-Raphson (IRLS) avec régularisation Ridge.

## Architecture du Répertoire
Pour que les flux de données entre R et C++ fonctionnent nativement, tous les fichiers doivent être placés dans le même dossier racine. 

```text
📁 Codes partie données Manning/
├── 📄 README.md
├── 📁 Eigen/                     # Bibliothèque mathématique C++ (inclue)
├── 📄 Manning_données_brutes.csv # Jeu de données au format csv
├── 📄 1_Preparation_Donnees.R    # Script R (Préparation et sélection BIC)
├── 📄 main.cpp                   # Code source C++ (Optimisation)
└── 📄 3_Analyse_Resultats.R      # Script R (Inférence et Graphes)
```

## Prérequis Techniques
* R (version 4.0+) avec les packages suivants installés : ggplot2, pROC.
* Compilateur C++ supportant la norme C++11 minimum (ex: g++ ou clang++).
* Bibliothèque Eigen3 (pour le calcul matriciel en C++). Note : Le dossier Eigen est déjà inclus dans cette archive, aucune installation supplémentaire n'est requise.

---

## Instructions d'Exécution (Reproductibilité)

L'exécution doit suivre un ordre strict pour respecter le flux de transfert des données entre R et C++. Placez votre terminal à la racine du dossier Codes partie données Manning/.

### Étape 1 : Préparation et Sélection de Modèles (R)
Ce script gère le nettoyage, la standardisation des variables continues, effectue une sélection de variables (Stepwise) basée sur le critère BIC, et exporte les matrices de design pour les trois modèles étudiés.

```bash
Rscript 1_Preparation_Donnees.R
```
*Fichiers générés : export_manning.csv, export_best_ratio.csv, export_best_brut.csv.*

### Étape 2 : Optimisation Numérique et Estimation (C++)
Attention : Pour reproduire les temps de calcul mentionnés dans le mémoire, il est impératif de compiler avec le flag d'optimisation maximale -O3. Il faut également lier la bibliothèque locale Eigen.

1. Compilation :
*(Utilisation du flag -I . pour lier le dossier local Eigen)*
```bash
g++ -std=c++11 -O3 -I . main.cpp -o solver_logistique
```

2. Exécution :
```bash
./solver_logistique
```
*Fichiers générés : beta_manning.csv, beta_best_ratio.csv, beta_best_brut.csv.*

### Étape 3 : Inférence Statistique et Évaluation (R)
Ce script récupère les coefficients optimaux calculés par le C++ pour reconstruire la matrice d'Information de Fisher. Il génère les tests de Wald, les tests de Rapport de Vraisemblance (LRT), vérifie la validité du C++ face à la référence R, et évalue la capacité de généralisation via une Validation Croisée (5-Fold CV).

```bash
Rscript 3_Analyse_Resultats.R
```
*Résultats : Affichage des p-valeurs et des métriques en console, rapport de validation logicielle, et génération des graphiques analytiques (Courbes ROC, Forest Plot, Frontière de Décision).*

## Notes aux Évaluateurs
* Graines aléatoires : Les générateurs pseudo-aléatoires (set.seed en R) sont fixés pour garantir la stricte reproductibilité des sous-échantillons de la validation croisée.
* Certification Logicielle : L'écart numérique entre les estimateurs natifs de R (glm) et l'algorithme IRLS pénalisé en C++ est évalué automatiquement à l'Étape 3 avec une tolérance stricte (1e-05), prouvant l'intégrité du moteur de calcul.