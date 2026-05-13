# ==============================================================================
# CHAPITRE 4 : ANALYSE STATISTIQUE, INFÉRENCE ET MACHINE LEARNING (R)
# ==============================================================================

# Chargement silencieux des bibliothèques nécessaires
suppressPackageStartupMessages(library(ggplot2)) # pour les graphiques avancés
suppressPackageStartupMessages(library(pROC))    # pour calculer l'AUC et tracer la courbe ROC
suppressPackageStartupMessages(library(caret))   # pour la création des plis de la validation croisée

graphics.off() # ferme toutes les fenêtres graphiques ouvertes et réinitialise les paramètres 

# ==============================================================================
# 1. CHARGEMENT DES DONNÉES ET RÉCUPÉRATION DES NOMS
# ==============================================================================
cat("\n--- 1. Chargement des donnees et des noms de variables ---\n")

# Chargement du jeu d'entraînement numérisé
df_train <- read.csv("export_adult_train.csv", header = FALSE) 
Y_train <- df_train[, 1]
X_train <- as.matrix(df_train[, -1])

# Chargement des coefficients optimaux exportés par C++
betas_df <- read.csv("beta_adult_train.csv", header = TRUE) 
beta_cpp <- as.numeric(betas_df$Coefficient)

# Récupération automatique des noms via model.matrix sur les données brutes
noms_cols <- c("age", "workclass", "fnlwgt", "education", "education_num", 
               "marital_status", "occupation", "relationship", "race", "sex", 
               "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")
df_t1 <- read.csv("adult.data", header = FALSE, col.names = noms_cols, strip.white = TRUE)
df_t2 <- read.csv("adult.test", header = FALSE, col.names = noms_cols, strip.white = TRUE, skip = 1)
df_noms <- rbind(df_t1, df_t2)
df_noms[df_noms == "?"] <- NA
df_noms <- na.omit(df_noms)

df_noms$education_num <- NULL

# Extraction des noms de colonnes (X_design contient l'Intercept en 1ère position)
X_design_temp <- model.matrix(~ . - income, data = df_noms)
noms_variables <- colnames(X_design_temp)
noms_variables[1] <- "Intercept" # renommage propre


# ==============================================================================
# 2. VALIDATION ALGORITHMIQUE (C++ VS R)
# ==============================================================================
cat("\n--- 2. Validation Algorithmique (Stabilite du C++) ---\n")

df_glm <- as.data.frame(cbind(Y = Y_train, X_train))
modele_R <- suppressWarnings(glm(Y ~ . - 1, data = df_glm, family = binomial(link = "logit")))
beta_R <- as.numeric(coef(modele_R)) 

comparaison_df <- data.frame(Variable = noms_variables, Beta_CPP = beta_cpp, Beta_R = beta_R, Diff = abs(beta_cpp - beta_R))
cat("Top 5 des divergences (Preuve de l'effet de la regularisation Ridge) :\n")
print(head(comparaison_df[order(-comparaison_df$Diff), ], 5)) 

plot(beta_R, beta_cpp, pch=16, col="darkred", main="Coefficients : R (Non-reg) vs C++ (Ridge)",
     xlab="R (glm)", ylab="C++ (IRLS)", xlim=c(-25, 5), ylim=c(-20, 5))
abline(0, 1, col="blue", lwd=2, lty=2)
grid()
dev.copy(png, "Graphe_1_Comparaison_Coefficients.png", width=800, height=600) # On copie ce qui est affiché à l'écran vers un fichier PNG
dev.off() # On ferme le fichier de destination pour valider l'enregistrement

# ==============================================================================
# 3. INFÉRENCE STATISTIQUE (FISHER & WALD)
# ==============================================================================
cat("\n--- 3. Inference Statistique (Fisher, Wald, LRT) ---\n") # affichage du titre de la section dans la console

eta <- X_train %*% beta_cpp # calcul du prédicteur linéaire 
P <- 1 / (1 + exp(-eta)) # calcul des probabilités estimées via la fonction logistique
P <- pmax(pmin(P, 1 - 1e-15), 1e-15) # troncature pour éviter l'instabilité numérique des logarithmes tendant vers l'infini

W_diag <- as.numeric(P * (1 - P)) # calcul de la variance locale de la loi de bernoulli
Information_Fisher <- t(X_train) %*% (W_diag * X_train) # construction de la matrice d'information de fisher asymptotique
I_penalisee <- Information_Fisher + diag(c(0, rep(1e-5, ncol(X_train)-1))) # intégration de la pénalité ridge en isolant l'intercept

Var_Cov <- solve(I_penalisee) # inversion de la matrice de fisher pour obtenir la matrice de variance-covariance
SE <- sqrt(diag(Var_Cov)) # extraction des écarts-types asymptotiques des estimateurs
Z_stat <- beta_cpp / SE # calcul de la statistique du test de wald
p_valeurs <- 2 * (1 - pnorm(abs(Z_stat))) # calcul de la p-valeur pour un test bilatéral

LogLik_Modele <- sum(Y_train * log(P) + (1 - Y_train) * log(1 - P)) # calcul de la log-vraisemblance du modèle optimisé
LogLik_Null <- sum(Y_train * log(mean(Y_train)) + (1 - Y_train) * log(1 - mean(Y_train))) # calcul de la log-vraisemblance du modèle restreint à la constante
cat(sprintf("LRT : Stat = %.2f | P-valeur = %e\n", 2*(LogLik_Modele - LogLik_Null), 
            pchisq(2*(LogLik_Modele - LogLik_Null), df=ncol(X_train)-1, lower.tail=FALSE))) # évaluation de la significativité globale par le test du chi-deux

# Critères d'Information (AIC & BIC)
k <- ncol(X_train) # nombre total de paramètres (p variables + 1 intercept)
n <- nrow(X_train) # nombre d'observations

AIC_val <- 2 * k - 2 * LogLik_Modele # Formule de l'AIC
BIC_val <- k * log(n) - 2 * LogLik_Modele # Formule du BIC

cat(sprintf("\nCriteres de Parcimonie du Modele :\n")) 
cat(sprintf(" -> AIC : %.2f\n", AIC_val)) 
cat(sprintf(" -> BIC : %.2f\n", BIC_val)) 

# Comparaison avec le modèle nul (sans aucune variable)
AIC_null <- 2 * 1 - 2 * LogLik_Null # calcul du critère aic de référence
BIC_null <- 1 * log(n) - 2 * LogLik_Null # calcul du critère bic de référence
cat(sprintf("\n(Pour comparaison, Modele Null -> AIC : %.2f | BIC : %.2f)\n", AIC_null, BIC_null)) # affichage des métriques comparatives


# ==============================================================================
# 4. PERFORMANCE SUR LE JEU DE TEST
# ==============================================================================
cat("\n--- 4. Validation Predictive (Jeu de test) ---\n")

df_test <- read.csv("export_adult_test.csv", header = FALSE) 
Y_test <- df_test[, 1]
X_test <- as.matrix(df_test[, -1])

P_pred_test <- 1 / (1 + exp(-(X_test %*% beta_cpp))) 
cat(sprintf("Accuracy globale : %.2f %%\n", (sum(diag(table(Y_test, ifelse(P_pred_test >= 0.5, 1, 0)))) / length(Y_test)) * 100))

roc_obj <- roc(Y_test, as.numeric(P_pred_test), quiet = TRUE) 
plot(roc_obj, col="darkblue", lwd=2, main=paste("Courbe ROC - AUC =", round(auc(roc_obj), 4))) # tracé de la courbe roc à l'écran
grid() # ajout de la grille de lecture
dev.copy(png, "Graphe_2_Courbe_ROC_Adult.png", width=800, height=600) # copie du graphique affiché vers le fichier png
dev.off() # fermeture du fichier pour valider l'enregistrement

# ==============================================================================
# 5. VALIDATION CROISÉE K-FOLD 
# ==============================================================================
cat("\n--- 5. Validation Croisee K-Fold (K=5) ---\n")

set.seed(42) 
K <- 5 
plis <- createFolds(Y_train, k = K, list = TRUE, returnTrain = FALSE) 

res_acc <- numeric(K) ; res_auc <- numeric(K) 

for(i in 1:K) {
  idx <- plis[[i]] 
  
  # Entraînement sur le pli courant
  m_fold <- suppressWarnings(glm(Y_train[-idx] ~ X_train[-idx, ] - 1, family = binomial)) 
  
  # R abandonne certaines variables (NA) à cause de la déficience de rang.
  # On remplace ces NA par 0 pour pouvoir faire la multiplication matricielle.
  coefs_R <- coef(m_fold)
  coefs_R[is.na(coefs_R)] <- 0 
  
  # Prédiction (avec as.numeric pour éviter l'avertissement de pROC)
  p_fold <- as.numeric(1 / (1 + exp(-(X_train[idx, ] %*% coefs_R)))) 
  
  # Calcul des métriques
  res_acc[i] <- sum(diag(table(Y_train[idx], ifelse(p_fold >= 0.5, 1, 0)))) / length(idx)
  res_auc[i] <- as.numeric(auc(roc(Y_train[idx], p_fold, quiet = TRUE)))
  
  # Affichage en temps réel
  cat(sprintf(" Pli %d : Accuracy = %.4f | AUC = %.4f\n", i, res_acc[i], res_auc[i]))
}

cat(sprintf("\n=> Moyenne CV Accuracy : %.2f %% | AUC : %.4f\n", mean(res_acc)*100, mean(res_auc)))

# Division de la fenêtre en 1 ligne et 2 colonnes
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2)) # configuration de l'affichage graphique en deux panneaux cote a cote

# 1. Graphique pour l'Accuracy
plot(1:K, res_acc, type = "b", pch = 16, col = "dodgerblue", lwd = 2,
     ylim = c(min(res_acc) - 0.01, max(res_acc) + 0.01),
     xaxt = "n", xlab = "Numéro du pli (Fold)", ylab = "Accuracy",
     main = "Évolution de l'Accuracy par pli", cex.main = 1.1) # trace de la courbe de l'accuracy pour chaque pli
axis(1, at = 1:K) # ajout manuel des graduations sur l'axe des abscisses
abline(h = mean(res_acc), col = "firebrick", lty = 2, lwd = 2) # ajout d'une ligne horizontale representant la moyenne globale
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted") # ajout de la grille de lecture en arriere-plan
legend("bottomleft", legend = sprintf("Moyenne : %.2f%%", mean(res_acc)*100),
       col = "firebrick", lty = 2, lwd = 2, bty = "n") # insertion de la legende affichant la valeur moyenne calculee

# 2. Graphique pour l'AUC
plot(1:K, res_auc, type = "b", pch = 16, col = "forestgreen", lwd = 2,
     ylim = c(min(res_auc) - 0.01, max(res_auc) + 0.01),
     xaxt = "n", xlab = "Numéro du pli (Fold)", ylab = "AUC",
     main = "Évolution de l'AUC par pli", cex.main = 1.1) # trace de la courbe de l'auc pour chaque pli
axis(1, at = 1:K) # ajout manuel des graduations sur l'axe des abscisses
abline(h = mean(res_auc), col = "firebrick", lty = 2, lwd = 2) # ajout d'une ligne horizontale representant l'auc moyenne globale
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted") # ajout de la grille de lecture en arriere-plan
legend("bottomleft", legend = sprintf("Moyenne : %.4f", mean(res_auc)),
       col = "firebrick", lty = 2, lwd = 2, bty = "n") # insertion de la legende affichant la valeur de l'auc moyenne

dev.copy(png, "Graphe_3_Validation_Croisee.png", width=900, height=450) # copie du double graphique affiche vers le fichier png
dev.off() # fermeture du fichier pour valider l'enregistrement
cat("-> Graphique 3 'Graphe_3_Validation_Croisee.png' exporté (Format Lignes/Points).\n") # confirmation de l'export dans la console

# ==============================================================================
# 6. ANALYSE D'INFLUENCE (5 FAVORISANTS, 5 PÉNALISANTS, 5 INSIGNIFIANTS)
# ==============================================================================
cat("\n--- 6. Analyse des 15 Variables Cles (Fav, Sup, Neutres) ---\n")

df_inf <- data.frame(Variable = noms_variables, Beta = beta_cpp, P = p_valeurs)
df_inf <- df_inf[df_inf$Variable != "Intercept", ] # retrait intercept pour analyse

# Sélection des 3 groupes de 5 variables
df_fav <- head(df_inf[order(df_inf$Beta, decreasing = TRUE), ], 5) # les 5 plus gros positifs
df_sup <- head(df_inf[order(df_inf$Beta, decreasing = FALSE), ], 5) # les 5 plus gros négatifs
df_neu <- head(df_inf[order(abs(df_inf$Beta)), ], 5) # les 5 plus proches de zéro (insignifiants)

cat("\n[1] Top 5 FAVORISANTS (>50K) :\n") ; print(df_fav)
cat("\n[2] Top 5 PENALISANTS (<50K) :\n") ; print(df_sup)
cat("\n[3] Top 5 INSIGNIFIANTS (Sans effet) :\n") ; print(df_neu)

# Préparation du Graphe 4 (Tornado Chart avec 15 variables)
df_plot <- rbind(df_fav, df_sup, df_neu)
df_plot <- df_plot[order(df_plot$Beta), ] 
df_plot$Variable <- factor(df_plot$Variable, levels = df_plot$Variable)

p <- ggplot(df_plot, aes(x = Beta, y = Variable, fill = ifelse(Beta > 0.1, "Pos", ifelse(Beta < -0.1, "Neg", "Neutre")))) + # initialisation du graphique avec les donnees et les couleurs conditionnelles
  geom_bar(stat = "identity", color = "black", alpha = 0.8) + # trace des barres horizontales avec bordures
  geom_vline(xintercept = 0, color = "black", linewidth = 1.2) + # ajout d'une ligne verticale de reference a zero
  scale_fill_manual(values = c("Pos" = "steelblue", "Neg" = "firebrick", "Neutre" = "gray70"),  # definition manuelle des couleurs
                    name = "Impact", labels = c("Pénalise", "Sans effet", "Favorise")) + # personnalisation des etiquettes de la legende
  scale_x_continuous(limits = c(-18, 18), breaks = seq(-15, 15, by = 5)) + # <-- FORCAGE DE LA SYMETRIE ICI
  theme_minimal() + # application d'un theme visuel epure
  labs(title = "Analyse des 15 variables cles (Favorisants, Suppresseurs et Neutres)", # definition du titre principal
       x = "Coefficient Beta", y = "Variables du dataset Adult") + # definition du nom des axes
  theme(axis.text.y = element_text(size = 11, face = "bold"), legend.position = "bottom") # ajustement de la typographie et de la position de la legende

print(p) # impression explicite pour éviter le graphe vide et affichage a l'ecran
dev.copy(png, "Graphe_4_Importance_Variables.png", width=900, height=800) # copie du graphique affiche vers le fichier png
dev.off() # fermeture du fichier pour valider l'enregistrement

cat("\n--- FIN DE L'ANALYSE : Tous les graphes sont exportes. ---\n") # confirmation de fin de script dans la console

# ==============================================================================
# 7. EXPORT EXHAUSTIF DES DIVERGENCES (R VS C++)
# ==============================================================================
cat("\n--- 7. Analyse exhaustive de la pollution Ridge sur les 97 variables ---\n")

df_complet <- data.frame(
  Variable = noms_variables,
  Beta_CPP = beta_cpp,
  Beta_R = beta_R,
  Difference_Absolue = abs(beta_cpp - beta_R),
  stringsAsFactors = FALSE
)

df_complet <- df_complet[order(-df_complet$Difference_Absolue), ]

cat("Analyse de l'integrite du modele :\n")
cat(sprintf("- Nombre total de variables : %d\n", nrow(df_complet)))
cat(sprintf("- Variables 'polluees' (Diff > 0.1) : %d\n", sum(df_complet$Difference_Absolue > 0.1, na.rm = TRUE)))
cat(sprintf("- Variables 'integres' (Diff < 1e-4) : %d\n", sum(df_complet$Difference_Absolue < 1e-4, na.rm = TRUE)))

cat("\nTop 10 des plus grandes differences (Impact de la separation) :\n")
print(head(df_complet, 10))

cat("\nTop 10 des plus petites differences (Preuve de non-pollution) :\n")
print(tail(df_complet, 10))