# ==============================================================================
# CHAPITRE 4 : PRÉPARATION DES DONNÉES DE BIOMÉTRIE (R)
# ==============================================================================

# 1. Chargement et Nettoyage des données
# ------------------------------------------------------------------------------
cat("--- Chargement des donnees ---\n")
df <- read.csv("Manning_données_brutes.csv", stringsAsFactors = FALSE)

# Remplacement des virgules par des points pour la conversion numérique
cols_mesures <- c("Taille.index", "Taille.majeur", "Taille.annulaire", "largeur.paume", "longueur.paume")
for(col in cols_mesures) { # je prends les éléments dans cols_mesures
  df[[col]] <- as.numeric(gsub(",", ".", df[[col]])) # substitue tous les virgules en points (gsub)
}                                                    # transforme les chaines de caractère en nombres (as.numeric)

df <- na.omit(df) # supprime toutes lignes avec des valeurs manquantes
df$Y <- ifelse(df$sexe == "H", 1, 0) # binarisation de la cible (Y = 1 si Homme, 0 si Femme)


# 2. Modèle 1 : L'Indice de Manning classique (2D:4D)
# ------------------------------------------------------------------------------
cat("--- 1. Modele Historique (Indice de Manning) ---\n")
df$Manning <- df$Taille.index / df$Taille.annulaire # création de l'indicde de manning dans le data frame


# 3. Modèle 2 : Recherche du meilleur Ratio (Variables proportionnelles)
# ------------------------------------------------------------------------------
cat("--- 2. Recherche du meilleur Ratio (Minimisation BIC) ---\n")
ratios_names <- c() # vecteur vide pour stocker les noms de ratio
for(i in 1:(length(cols_mesures)-1)) {
  for(j in (i+1):length(cols_mesures)) { # nouvel indice pour croiser les données qu'une seule fois
    name <- paste0("Ratio_", i, "_", j) # crée une étiquette de texte ex; ratio_1_2 si on divise 1 par 2
    df[[name]] <- df[[cols_mesures[i]]] / df[[cols_mesures[j]]] # calcul vectorisé du rapport pour tous les individus
    ratios_names <- c(ratios_names, name) # ajoute du ratio au vecteur de noms de ratio (fait 10 fois)
  }
}

best_bic_ratio <- Inf # crée une varibale qui vaut l'infini pour stocker le meilleur bic trouvé (minimise le bic)
best_ratio_var <- "" # crée une chaine de caractère vide

# Test exhaustif de chaque ratio
for(var in ratios_names) { # on prend les 10 ratios qu'on a calculé
  modele <- glm(Y ~ df[[var]], family = binomial(link = "logit"), data = df) # on fait la régression logistique du modele avec glm
  if(BIC(modele) < best_bic_ratio) { # on regarde si le bic du modèle est le plus petit
    best_bic_ratio <- BIC(modele) 
    best_ratio_var <- var
  }
}
cat("Meilleur Ratio trouve :", best_ratio_var, "avec BIC =", best_bic_ratio, "\n")


# 4. Modèle 3 : Recherche du meilleur modèle sur variables initiales brutes
# ------------------------------------------------------------------------------
cat("--- 3. Recherche du modele optimal sur variables brutes (Stepwise BIC) ---\n")

# Définition du modèle vide et du modèle contenant les 5 variables initiales
modele_vide <- glm(Y ~ 1, family = binomial(link = "logit"), data = df) # on fait la régression logistique du modele sans variable explicative
formule_brutes <- as.formula(paste("Y ~", paste(cols_mesures, collapse=" + "))) # creéation de l'équation Y ~ Taille.index + Taille.majeur + Taille.annulaire + largeur.paume + longueur.paume
modele_sature_brutes <- glm(formule_brutes, family = binomial(link = "logit"), data = df) # régression avec toutes les variables

# Algorithme Stepwise avec pénalité BIC (k = log(n))
# l'algorithme peut ajouter une variable si elle baisse le bic mais aussi retirer celles qui avaient été ajoutées avant si on a colinéarité
modele_optimal_brut <- step(modele_vide, 
                            scope = list(lower = modele_vide, upper = modele_sature_brutes), # donne les bornes du modèle voulu
                            direction = "both", # a le droit d'ajouter ou de supprimer des variables  
                            k = log(nrow(df)), # par défaut, stepwise utilise l'AIC donc on transforme la fonction pour qu'elle optimise le BIC
                            trace = 0) # Rend silencieux l'écriture dans la console

best_bic_brut <- BIC(modele_optimal_brut)
best_vars_brutes <- names(coef(modele_optimal_brut))[-1] # Exclusion du modèle sans variables 

cat("Meilleur modele (variables brutes) trouve avec BIC =", best_bic_brut, "\n")
cat("Variables retenues :", paste(best_vars_brutes, collapse=", "), "\n")


# 5. Standardisation et Exportation des trois matrices de Design (X)
# ------------------------------------------------------------------------------
cat("--- Standardisation et Exportation ---\n")
# L'utilisation de drop=FALSE garantit que R conserve la structure matricielle (pas de tranformation en vecteur)
X_manning <- scale(df[, "Manning", drop=FALSE]) #  on centre et on réduit les données
X_best_ratio <- scale(df[, best_ratio_var, drop=FALSE])
X_best_brut <- scale(df[, best_vars_brutes, drop=FALSE])

# Ajout de la colonne Intercept (des 1)
X_manning_design <- cbind(Intercept = 1, X_manning) # ajout de l'intercept 
X_best_ratio_design <- cbind(Intercept = 1, X_best_ratio)
X_best_brut_design <- cbind(Intercept = 1, X_best_brut)

# Exportation des 3 jeux de données
write.table(cbind(Y = df$Y, X_manning_design), "export_manning.csv", sep=",", row.names=FALSE, col.names=FALSE)
write.table(cbind(Y = df$Y, X_best_ratio_design), "export_best_ratio.csv", sep=",", row.names=FALSE, col.names=FALSE)
write.table(cbind(Y = df$Y, X_best_brut_design), "export_best_brut.csv", sep=",", row.names=FALSE, col.names=FALSE)

cat("Les 3 jeux de donnees biometriques sont prets pour le moteur C++ !\n")
