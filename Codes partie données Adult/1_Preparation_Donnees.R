# ==============================================================================
# CHAPITRE 4 : PRÉPARATION DES DONNÉES ADULT CENSUS (R)
# ==============================================================================

# 1. Définition des colonnes et Chargement des données

cat("--- Chargement des donnees ---\n")

# D'après le fichier adult.names, on liste manuellement les colonnes
noms_colonnes <- c("age", "workclass", "fnlwgt", "education", "education_num", 
                   "marital_status", "occupation", "relationship", "race", "sex", 
                   "capital_gain", "capital_loss", "hours_per_week", "native_country", "income")

# Chargement du jeu d'entraînement (strip.white enlève les espaces cachés au début des mots)
df_train <- read.csv("adult.data", header = FALSE, col.names = noms_colonnes, strip.white = TRUE)
df_train$is_train <- TRUE # on ajoute un marqueur pour le retrouver plus tard



# Chargement du jeu de test (skip = 1 pour ignorer la première ligne "|1x3 Cross validator")
df_test <- read.csv("adult.test", header = FALSE, col.names = noms_colonnes, strip.white = TRUE, skip = 1)
df_test$is_train <- FALSE # marqueur test

# Fusion temporaire pour que le One-Hot Encoding crée exactement les mêmes colonnes partout
df <- rbind(df_train, df_test) 

df$education_num <- NULL # SUPPRESSION DE LA VARIABLE REDONDANTE

# 2. Nettoyage et Binarisation de la Cible

cat("--- Nettoyage et binarisation ---\n")

# Dans ce dataset, les valeurs manquantes sont notées "?"
df[df == "?"] <- NA # remplace les "?" par de vrais NA reconnus par R
df <- na.omit(df)   # supprime toutes les lignes avec des valeurs manquantes

# Binarisation de la cible (Y = 1 si >50K, 0 sinon)
# Attention : dans adult.test, les labels ont un point à la fin (">50K."). grepl gère ça facilement.
df$Y <- ifelse(grepl(">50K", df$income), 1, 0) 

df$income <- NULL # On retire la colonne texte d'origine pour ne garder que notre cible numérique Y


# 3. Standardisation des variables continues

cat("--- Standardisation des variables continues ---\n")

# On isole les variables qui sont des nombres continus
cols_continues <- c("age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week")

for(col in cols_continues) { # je prends les éléments dans cols_continues
  df[[col]] <- scale(df[[col]]) # on centre et on réduit la donnée
}


# 4. One-Hot Encoding (Variables Muettes)

cat("--- Creation de la Matrice d'Experience (One-Hot Encoding) ---\n")

# model.matrix transforme automatiquement le texte en colonnes 0/1. 
# elle supprime automatiquement la 1ère modalité (K-1) 
# pour éviter la colinéarité parfaite (Hypothèse de la matrice X de plein rang garantie)
# Le "~ . - Y - is_train" signifie : utilise toutes les variables sauf Y et is_train
X_design <- model.matrix(~ . - Y - is_train, data = df)

# Note : model.matrix ajoute d'elle-même une colonne "Intercept" remplie de 1.
# Pas besoin de faire un cbind(Intercept = 1, ...) comme dans l'ancien script 


# 5. Séparation Train/Test et Exportation

cat("--- Separation et Exportation ---\n")

# On sépare à nouveau en utilisant notre marqueur
X_train <- X_design[df$is_train == TRUE, ]
Y_train <- df$Y[df$is_train == TRUE]

X_test <- X_design[df$is_train == FALSE, ]
Y_test <- df$Y[df$is_train == FALSE]

# Exportation des jeux de données finaux
write.table(cbind(Y = Y_train, X_train), "export_adult_train.csv", sep=",", row.names=FALSE, col.names=FALSE)
write.table(cbind(Y = Y_test, X_test), "export_adult_test.csv", sep=",", row.names=FALSE, col.names=FALSE)

cat("Termine. Les fichiers 'export_adult_train.csv' et 'export_adult_test.csv' ont ete crees avec succes.\n")

# Affichage de certaines variables 
# Rappel : Var_1 = colonne 2, donc Var_21 = colonne 22 de X_design
cat("La variable Var_21 est :", colnames(X_design)[22], "\n")
cat("La variable Var_70 est :", colnames(X_design)[71], "\n")