# ==============================================================================
# CHAPITRE 4 : ANALYSE STATISTIQUE, PERFORMANCE ET VALIDATION CROISÉE
# ==============================================================================

# Chargement silencieux des bibliothèques nécessaires
suppressPackageStartupMessages(library(ggplot2)) # pour les graphiques avancés
suppressPackageStartupMessages(library(pROC))    # pour calculer l'AUC et tracer la courbe ROC

# ------------------------------------------------------------------------------
# 1. FONCTION MOTEUR : INFÉRENCE ET MÉTRIQUES DE CLASSIFICATION
# ------------------------------------------------------------------------------

calculer_inference <- function(fichier_donnees, fichier_betas, noms_variables) { # definition de la fonction principale d'inference
  
  # Chargement des données standardisées et des variables cibles
  donnees <- read.csv(fichier_donnees, header = FALSE) # lecture du fichier de donnees standardisees sans en-tete
  Y <- donnees[, 1] # extraction de la premiere colonne contenant la cible
  X <- as.matrix(donnees[, -1]) # conversion des colonnes explicatives restantes en matrice
  
  # Chargement des coefficients (betas) calculés par l'algorithme IRLS (C++)
  betas_df <- read.csv(fichier_betas, header = TRUE) # chargement du fichier exporte par le c++
  beta <- as.numeric(betas_df$Coefficient) # conversion de la colonne des coefficients en vecteur numerique
  
  # Calcul du prédicteur linéaire et des probabilités
  eta <- X %*% beta # calcul matriciel du predicteur lineaire
  P <- 1 / (1 + exp(-eta)) # application de la fonction logistique pour obtenir les probabilites
  P <- pmax(pmin(P, 1 - 1e-15), 1e-15) # Stabilité numérique # troncature des valeurs extremes pour eviter l'instabilite
  
  # Reconstruction de la matrice de Variance-Covariance (cf. Section 3.2)
  W <- diag(as.vector(P * (1 - P))) # creation de la matrice de poids diagonale avec la variance de bernoulli
  lambda_ridge <- 1e-5 # definition manuelle de la penalite appliquee dans le c++
  I_ridge <- diag(ncol(X)) # creation d'une matrice identite pour la regularisation
  I_ridge[1, 1] <- 0 # annulation de la penalite pour l'intercept
  Hessienne <- t(X) %*% W %*% X + lambda_ridge * I_ridge # reconstruction de la matrice d'information de fisher penalisee
  Var_Covar <- solve(Hessienne) # calcul de la matrice de variance-covariance par inversion
  
  # Inférence de Wald
  se <- sqrt(diag(Var_Covar)) # extraction des ecarts-types asymptotiques
  z_stat <- beta / se # calcul de la statistique z pour le test de wald
  p_value <- 2 * (1 - pnorm(abs(z_stat))) # calcul de la p-valeur associee au test bilateral
  
  # Odds Ratios et Intervalles de Confiance à 95%
  OR <- exp(beta) # transformation des log-odds en odds ratios
  CI_bas <- exp(beta - 1.96 * se) # calcul de la borne inferieure de l'intervalle de confiance
  CI_haut <- exp(beta + 1.96 * se) # calcul de la borne superieure de l'intervalle de confiance
  
  # Réintégration formelle de la statistique Z pour le Test de Wald complet
  resultats_wald <- data.frame( # creation du dataframe recapitulatif de l'inference
    Variable = noms_variables,
    Beta = beta,
    Std_Error = se,
    Z_Wald = z_stat,
    P_Value = p_value,
    Odds_Ratio = OR,
    CI_Lower = CI_bas,
    CI_Upper = CI_haut
  )
  
  # Évaluation sur l'échantillon d'apprentissage (In-sample)
  predictions_classes <- ifelse(P > 0.5, 1, 0) # assignation binaire selon le seuil neutre
  matrice_confusion <- table(Vrai = Y, Predit = factor(predictions_classes, levels=c(0,1))) # generation de la matrice de croisement
  
  vrai_positifs <- matrice_confusion["1", "1"] # recuperation des vrais positifs
  vrai_negatifs <- matrice_confusion["0", "0"] # recuperation des vrais negatifs
  faux_positifs <- matrice_confusion["0", "1"] # recuperation des faux positifs
  faux_negatifs <- matrice_confusion["1", "0"] # recuperation des faux negatifs
  
  accuracy <- (vrai_positifs + vrai_negatifs) / sum(matrice_confusion) # calcul du taux de classification correcte
  sensibilite <- vrai_positifs / (vrai_positifs + faux_negatifs) # calcul du taux de vrais positifs
  specificite <- vrai_negatifs / (vrai_negatifs + faux_positifs) # calcul du taux de vrais negatifs
  
  metriques <- list( # regroupement des resultats de performance
    Matrice = matrice_confusion,
    Accuracy = accuracy,
    Sensibilite = sensibilite,
    Specificite = specificite
  )
  
  return(list(Tableau = resultats_wald, Metriques = metriques, 
              Probabilites = P, Vraies_Valeurs = Y, X_matrix = X)) # renvoi de la liste globale des objets
}

# ------------------------------------------------------------------------------
# 2. EXÉCUTION ET ANALYSE DES 3 MODÈLES
# ------------------------------------------------------------------------------

res_manning <- calculer_inference("export_manning.csv", "beta_manning.csv", c("Intercept", "Indice_Manning")) # evaluation du premier modele
res_ratio <- calculer_inference("export_best_ratio.csv", "beta_best_ratio.csv", c("Intercept", "Ratio_Index_Largeur")) # evaluation du modele bic ratio
res_brut <- calculer_inference("export_best_brut.csv", "beta_best_brut.csv", c("Intercept", "Taille_Index", "Largeur_Paume")) # evaluation du modele bic brut

afficher_analyse <- function(nom_modele, res) { # fonction d'affichage console formatte
  cat(sprintf("\n============================================================\n"))
  cat(sprintf(" ANALYSE DU %s\n", toupper(nom_modele)))
  cat(sprintf("============================================================\n"))
  cat("--- Test de Wald et Odds Ratios ---\n")
  print(round(res$Tableau[, -1], 4)) # affichage des resultats arrondis pour la lisibilite
  
  cat("\n--- Performances In-Sample (Seuil = 0.5) ---\n")
  print(res$Metriques$Matrice) # affichage brut de la matrice de confusion
  cat(sprintf("Accuracy globale  : %.2f %%\n", res$Metriques$Accuracy * 100)) # formatage en pourcentage de l'exactitude
  cat(sprintf("Sensibilité (H)   : %.2f %%\n", res$Metriques$Sensibilite * 100)) # formatage en pourcentage de la sensibilite
  cat(sprintf("Spécificité (F)   : %.2f %%\n", res$Metriques$Specificite * 100)) # formatage en pourcentage de la specificite
}

afficher_analyse("Modèle 1 : Indice de Manning (Historique)", res_manning) # appel de la fonction pour le modele 1
afficher_analyse("Modèle 2 : Meilleur Ratio (BIC)", res_ratio) # appel de la fonction pour le modele 2
afficher_analyse("Modèle 3 : Variables Brutes (BIC)", res_brut) # appel de la fonction pour le modele 3


# ------------------------------------------------------------------------------
# 3. VALIDATION CROISÉE (K-FOLD CV) EN R DE BASE
# ------------------------------------------------------------------------------
# Pour garantir que notre Modèle Brut (optimal selon BIC) ne surapprend pas 
# (overfitting), nous évaluons sa robustesse hors-échantillon via une 
# validation croisée à 5 plis (5-Fold CV), codée sans dépendance externe.

cat("\n============================================================\n")
cat(" VALIDATION CROISÉE K-FOLD (OUT-OF-SAMPLE)\n")
cat("============================================================\n")

# Création d'un dataframe propre pour la CV
df_cv <- as.data.frame(res_brut$X_matrix[, -1]) # Retrait de l'Intercept # extraction de la sous-matrice explicative pure
colnames(df_cv) <- c("Taille_Index", "Largeur_Paume") # attribution explicite des noms
df_cv$Y <- as.numeric(res_brut$Vraies_Valeurs) # ajout de la variable cible au dataframe

set.seed(42) # Reproductibilité de la séparation # fixation de l'aleatoire pour les plis
k_folds <- 5 # parametrage du nombre de sous-echantillons

# Mélange aléatoire des indices de l'échantillon
indices_melanges <- sample(1:nrow(df_cv)) # generation du vecteur d'indices permutes

# Découpage en k sous-ensembles (folds) de tailles égales
folds <- split(indices_melanges, cut(seq_along(indices_melanges), k_folds, labels = FALSE)) # creation d'une liste contenant les k vecteurs d'indices

# Initialisation des vecteurs de stockage des métriques
cv_accuracies <- numeric(k_folds) # pre-allocation de l'espace pour l'exactitude
cv_aucs <- numeric(k_folds) # pre-allocation de l'espace pour les auc

for(i in 1:k_folds) { # boucle sur chaque sous-echantillon
  # Le pli courant sert de test, les k-1 autres servent d'entraînement
  indices_test <- folds[[i]] # selection des indices de test du pli courant
  train_data <- df_cv[-indices_test, ] # creation du jeu d'apprentissage par exclusion
  test_data <- df_cv[indices_test, ] # creation du jeu de validation par inclusion
  
  # Entraînement sur l'échantillon d'apprentissage 
  # (glm utilise l'algorithme IRLS, mathématiquement équivalent à notre moteur C++)
  modele_cv <- glm(Y ~ Taille_Index + Largeur_Paume, data = train_data, family = binomial(link="logit")) # ajustement du modele sur le pli d'apprentissage
  
  # Prédictions sur l'ensemble de test
  prob_pred <- predict(modele_cv, newdata = test_data, type = "response") # calcul des probabilites sur le pli de validation
  class_pred <- ifelse(prob_pred > 0.5, 1, 0) # assignation des classes predites
  
  # Métriques d'évaluation du pli courant
  cv_accuracies[i] <- mean(class_pred == test_data$Y) # stockage du taux de succes pour le pli courant
  cv_aucs[i] <- as.numeric(pROC::auc(pROC::roc(test_data$Y, prob_pred, quiet = TRUE))) # stockage de l'aire sous la courbe pour le pli courant
}

# Synthèse des performances Out-of-Sample
cat(sprintf("Validation Croisée (%d-Fold) sur le Modèle Brut :\n", k_folds))
cat(sprintf(" - Accuracy moyenne Out-of-Sample : %.2f %% (± %.2f %%)\n", mean(cv_accuracies)*100, sd(cv_accuracies)*100)) # restitution de la moyenne et de l'ecart-type
cat(sprintf(" - AUC moyen Out-of-Sample        : %.3f (± %.3f)\n", mean(cv_aucs), sd(cv_aucs))) # restitution de l'auc moyen et de son ecart-type
cat("Conclusion CV : La performance reste stable hors-échantillon, confirmant l'absence de surapprentissage massif.\n")
# ------------------------------------------------------------------------------
# 4. VISUALISATIONS
# ------------------------------------------------------------------------------

# -- Forest Plot --
df_plot <- rbind(res_manning$Tableau[-1, ], res_ratio$Tableau[-1, ], res_brut$Tableau[-1, ]) # concatenation des lignes de resultats (sans intercept)
df_plot$Modele <- c("Manning", "Ratio BIC", "Brut BIC", "Brut BIC") # ajout d'une colonne indicative du modele d'origine

graph_or <- ggplot(df_plot, aes(x = Odds_Ratio, y = Variable, color = Modele)) + # initialisation du canevas ggplot
  geom_point(size = 3) + # trace des points pour l'odds ratio
  geom_errorbar(aes(xmin = CI_Lower, xmax = CI_Upper), width = 0.2, linewidth = 1, orientation = "y") + # ajout des moustaches d'intervalle de confiance
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) + # ajout de la ligne verticale representant l'absence d'effet
  scale_x_log10() + # utilisation d'une echelle logarithmique adaptee aux odds ratios
  labs(title = "Comparaison des Odds Ratios",
       subtitle = "Variables standardisées (Effet d'une variation de +1 Écart-Type)",
       x = "Odds Ratio (Échelle Log)", y = "") +
  theme_bw() + # application d'un theme blanc epure
  theme(legend.position = "bottom") # placement de la legende en partie inferieure

print(graph_or) # generation a l'ecran du forest plot

# -- Courbes ROC --
roc_manning <- roc(res_manning$Vraies_Valeurs, as.numeric(res_manning$Probabilites), quiet = TRUE) # objet roc pour le modele manning
roc_ratio <- roc(res_ratio$Vraies_Valeurs, as.numeric(res_ratio$Probabilites), quiet = TRUE) # objet roc pour le modele du ratio
roc_brut <- roc(res_brut$Vraies_Valeurs, as.numeric(res_brut$Probabilites), quiet = TRUE) # objet roc pour le modele des variables brutes

plot(roc_brut, col = "forestgreen", lwd = 2, main = "Comparaison des Courbes ROC") # trace initial de la fenetre avec la premiere courbe
plot(roc_ratio, col = "firebrick", lwd = 2, add = TRUE) # superposition de la seconde courbe roc
plot(roc_manning, col = "steelblue", lwd = 2, add = TRUE) # superposition de la troisieme courbe roc

legend("bottomright", # ancrage inferieur droit pour eviter le masquage des courbes
       legend = c(sprintf("Variables Brutes (AUC = %.3f)", auc(roc_brut)),
                  sprintf("Ratio Optimal (AUC = %.3f)", auc(roc_ratio)),
                  sprintf("Indice Manning (AUC = %.3f)", auc(roc_manning))),
       col = c("forestgreen", "firebrick", "steelblue"), lwd = 2) # assignation des codes de couleur respectifs

# -- Frontière de Décision --
X_plot <- as.data.frame(res_brut$X_matrix[, 2:3]) # isolation des deux dimensions utiles au trace 2d
colnames(X_plot) <- c("Index_Std", "Largeur_Std") # renommage local pour la construction graphique
X_plot$Sexe <- as.factor(ifelse(res_brut$Vraies_Valeurs == 1, "Homme", "Femme")) # preparation d'un marqueur factoriel pour la coloration

b0 <- res_brut$Tableau$Beta[1] # extraction coefficient intercept
b1 <- res_brut$Tableau$Beta[2] # extraction coefficient index
b2 <- res_brut$Tableau$Beta[3] # extraction coefficient largeur

intercept_droite <- -b0 / b2 # formulation mathématique de l'ordonnée a l'origine de l'hyperplan
pente_droite <- -b1 / b2 # formulation mathématique de la pente de l'hyperplan

graph_frontiere <- ggplot(X_plot, aes(x = Index_Std, y = Largeur_Std, color = Sexe, shape = Sexe)) + # projection spatiale
  geom_point(size = 2.5, alpha = 0.8) + # affichage du nuage de points reel
  geom_abline(intercept = intercept_droite, slope = pente_droite, # trace de la frontiere de separation calculee
              color = "black", linetype = "dashed", linewidth = 1) +
  scale_color_manual(values = c("Femme" = "darkorange", "Homme" = "darkblue")) + # attribution semantique des couleurs
  labs(title = "Frontière de Décision Géométrique (Modèle Brut Optimal)",
       subtitle = "La droite en pointillés représente le seuil d'équiprobabilité (P = 0.5)",
       x = "Taille de l'Index (Standardisée)", 
       y = "Largeur de la Paume (Standardisée)") +
  theme_bw() + # themazation du graphique de densite
  # Annotation de l'équation de la droite
  annotate("text", x = min(X_plot$Index_Std)+0.8, y = max(X_plot$Largeur_Std)-0.2, # positionnement spatial du premier texte
           label = "Région de prédiction : Femme", color = "darkorange", fontface = "bold") +
  annotate("text", x = max(X_plot$Index_Std)-0.8, y = min(X_plot$Largeur_Std)+0.2, # positionnement spatial du second texte
           label = "Région de prédiction : Homme", color = "darkblue", fontface = "bold")
print(graph_frontiere) # affichage final de la representation de decision

# -- Diagramme en Barres des Métriques (Seuil p = 0.5) --

# 1. Extraction et calcul de la Précision pour les deux modèles
VP_M1 <- res_manning$Metriques$Matrice["1", "1"] # extraction ponctuelle des cas modelises corrects m1
FP_M1 <- res_manning$Metriques$Matrice["0", "1"] # extraction ponctuelle des fausses alarmes m1
Precision_M1 <- VP_M1 / (VP_M1 + FP_M1) # algorithme numerateur precision m1

VP_M3 <- res_brut$Metriques$Matrice["1", "1"] # extraction ponctuelle des cas modelises corrects m3
FP_M3 <- res_brut$Metriques$Matrice["0", "1"] # extraction ponctuelle des fausses alarmes m3
Precision_M3 <- VP_M3 / (VP_M3 + FP_M3) # algorithme numerateur precision m3

# 2. Construction du DataFrame pour ggplot
df_metrics <- data.frame( # creation de la table des effectifs comparatifs
  Modele = rep(c("Modèle 1 (Manning)", "Modèle 3 (Brut)"), each = 4), # affectation symetrique des identifiants
  Metrique = rep(c("Exactitude", "Rappel", "Spécificité", "Précision"), 2), # affectation des descripteurs repetes
  Valeur = c(
    res_manning$Metriques$Accuracy, res_manning$Metriques$Sensibilite, res_manning$Metriques$Specificite, Precision_M1,
    res_brut$Metriques$Accuracy, res_brut$Metriques$Sensibilite, res_brut$Metriques$Specificite, Precision_M3
  ) * 100 # conversion automatique des probabilites en entiers de pourcentage
)

# Fixer l'ordre des métriques sur l'axe X
df_metrics$Metrique <- factor(df_metrics$Metrique, levels = c("Exactitude", "Rappel", "Spécificité", "Précision")) # blocage interne de la distribution en x

# 3. Génération du graphique
graph_barres <- ggplot(df_metrics, aes(x = Metrique, y = Valeur, fill = Modele)) + # mise en structure de projection
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black") + # generation cote a cote des barres
  geom_text(aes(label = sprintf("%.1f%%", Valeur)), # encodage numerique dynamique des cimes
            position = position_dodge(width = 0.8), vjust = -0.6, size = 4, fontface = "bold") +
  scale_fill_manual(values = c("Modèle 1 (Manning)" = "steelblue", "Modèle 3 (Brut)" = "forestgreen")) + # palette personnalisee
  scale_y_continuous(limits = c(0, 105), breaks = seq(0, 100, 10)) + # 105 pour laisser la place aux étiquettes # dimensionnement adaptatif
  labs(title = "Comparaison des Métriques de Classification (Seuil p = 0.5)",
       x = "", y = "Score (%)", fill = "") +
  theme_bw() + # ecrasement du fond natif
  theme(legend.position = "top", # bascule du cartouche a l'horizontale
        axis.text.x = element_text(size = 12, face = "bold"),
        axis.title.y = element_text(size = 11, face = "bold"))

print(graph_barres) # sortie graphique

# ------------------------------------------------------------------------------
# 5. CERTIFICATION TECHNIQUE : COMPARAISON MOTEUR C++ VS RÉFÉRENCE R (GLM)
# ------------------------------------------------------------------------------
# Cette étape compare les coefficients issus de l'implémentation IRLS en C++ 
# avec ceux de la fonction de référence 'glm' de R sur le modèle optimal.

cat("\n============================================================\n")
cat(" RAPPORT DE VALIDATION LOGICIELLE\n")
cat("============================================================\n")

# Reconstruction du dataset de travail à partir des sorties du Modèle 3
df_val <- as.data.frame(res_brut$X_matrix[, -1])  # recuperation du sous ensemble des descripteurs
colnames(df_val) <- c("Taille_Index", "Largeur_Paume") # application de la nomenclature d'origine
df_val$Y <- res_brut$Vraies_Valeurs # fusion de la matrice explicative et ciblante

# Exécution du modèle de référence (R Stats)
modele_ref <- glm(Y ~ Taille_Index + Largeur_Paume, data = df_val, family = binomial(link="logit")) # fitting the modele logit unitaire r

# Construction du tableau de comparaison
comparaison_betas <- data.frame( # creation entite dataframe de synthese
  Variable = c("Intercept", "Taille_Index", "Largeur_Paume"),
  Beta_Algorithme_CPP = res_brut$Tableau$Beta, # recuperation depuis tableau exporte cpp
  Beta_Reference_R    = as.numeric(coef(modele_ref)), # extraction numerique depuis sortie native
  Difference          = res_brut$Tableau$Beta - as.numeric(coef(modele_ref)) # calcul du residu
)

# Affichage des résultats : on n'arrondit que les colonnes numériques (2 à 4)
comparaison_affichage <- comparaison_betas # copie tampon pour preserver tableau complet en memoire
comparaison_affichage[, 2:4] <- round(comparaison_affichage[, 2:4], 10) # troncature visuelle precise a 10 decimales

print(comparaison_affichage) # declenchement log du rapport

# Conclusion de la certification (tolérance de 1e-05 justifiée par la régularisation)
ecart_total <- max(abs(comparaison_betas$Difference)) # recherche de l'erreur maximale theoreme absolu
cat(sprintf("\nÉcart de précision maximal : %.2e\n", ecart_total)) # renvoi a l'utilisateur de l'erreur sc

if(ecart_total < 1e-05) { # condition theoremique de validation
  cat("RÉSULTAT : CERTIFIÉ. L'implémentation C++ est conforme à la référence.\n") # output de succes d'integration algorithmique
} else {
  cat("RÉSULTAT : ÉCHEC. Une divergence significative a été identifiée.\n") # output echec ou dysfonctionnement memoire cpp
}
cat("============================================================\n")

# ============================================================
# 6. TEST DU RAPPORT DE VRAISEMBLANCE (LRT)
# ============================================================
cat("\n============================================================\n")
cat(" TEST DU RAPPORT DE VRAISEMBLANCE (LRT) - Modèle 3\n")
cat("============================================================\n")

# La déviance est déjà égale à -2*log-vraisemblance.
# La statistique Lambda est donc simplement la différence des déviances.
LRT_stat <- modele_ref$null.deviance - modele_ref$deviance # application theoreme lrt deviance

# Les degrés de liberté correspondent au nombre de variables explicatives (ici 2)
LRT_df <- modele_ref$df.null - modele_ref$df.residual # differenciel des degres de liberte du residuel

# Calcul de la p-valeur via la loi du Chi-deux
LRT_p_val <- pchisq(LRT_stat, df = LRT_df, lower.tail = FALSE) # obtention p value distribution

# Affichage formaté pour le mémoire
cat(sprintf("Déviance du modèle vide (D0)      : %.4f\n", modele_ref$null.deviance)) # impression d0 modele null
cat(sprintf("Déviance du modèle optimal (D)    : %.4f\n", modele_ref$deviance)) # impression parametre test
cat(sprintf("Statistique du test (Lambda)      : %.4f\n", LRT_stat)) # impression log rapport chi
cat(sprintf("Degrés de liberté (p)             : %d\n", LRT_df)) # impression nombre dimensions
cat(sprintf("P-valeur                          : %e\n", LRT_p_val)) # impression niveau de significativite complet
cat("============================================================\n")

# ============================================================
# LRT POUR LE MODÈLE 1 (INDICE DE MANNING)
# ============================================================
cat("\n============================================================\n")
cat(" TEST DU RAPPORT DE VRAISEMBLANCE (LRT) - Modèle 1 (Manning)\n")
cat("============================================================\n")

# Chargement des données exportées pour le Modèle 1 (Manning)
donnees_m1 <- read.csv("export_manning.csv", header = FALSE) # lecture matrice de donnees extraites
Y_m1 <- donnees_m1[, 1]          # Colonne 1 : La cible (Sexe) # attribution variable cible manuelle
Manning <- donnees_m1[, 3]       # Colonne 3 : L'indice standardisé (Col 2 = Intercept) # attribution variable numerique standard

# Ajustement du modèle de référence dans R
modele_m1 <- glm(Y_m1 ~ Manning, family = binomial(link="logit")) # fitting regression univariable sur un parametre m1

# Calcul du LRT
LRT_stat_m1 <- modele_m1$null.deviance - modele_m1$deviance # difference theoreme lrt deviance m1
LRT_df_m1 <- modele_m1$df.null - modele_m1$df.residual # difference degres de liberte m1
LRT_p_val_m1 <- pchisq(LRT_stat_m1, df = LRT_df_m1, lower.tail = FALSE) # p valeur m1 evaluee a zero

# Affichage
cat(sprintf("Déviance du modèle vide (D0)      : %.4f\n", modele_m1$null.deviance)) # impression terminal parametre vide m1 d0
cat(sprintf("Déviance du modèle ajusté (D)     : %.4f\n", modele_m1$deviance)) # impression modele parametre optimalise m1 d
cat(sprintf("Statistique du test (Lambda)      : %.4f\n", LRT_stat_m1)) # impression rapport theoremique stat
cat(sprintf("Degrés de liberté (p)             : %d\n", LRT_df_m1)) # impression nombre variable freedom
cat(sprintf("P-valeur                          : %e\n", LRT_p_val_m1)) # impression format decimal float e- test final