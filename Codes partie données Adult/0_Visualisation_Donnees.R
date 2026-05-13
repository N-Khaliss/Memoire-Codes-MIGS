# ==============================================================================
# EXPLORATION DU JEU DE DONNÉES ADULT CENSUS INCOME
# ==============================================================================

# 1. Définition des noms de colonnes (extraits du fichier adult.names)
noms_colonnes <- c(
  "age", 
  "workclass", 
  "fnlwgt", 
  "education", 
  "education_num", 
  "marital_status", 
  "occupation", 
  "relationship", 
  "race", 
  "sex", 
  "capital_gain", 
  "capital_loss", 
  "hours_per_week", 
  "native_country", 
  "income" # La variable cible (>50K ou <=50K)
)

# 2. Chargement du fichier de données brutes
# - header = FALSE : le fichier adult.data n'a pas de ligne d'en-tête
# - col.names : on assigne directement notre vecteur de noms
# - strip.white = TRUE : supprime les espaces parasites après les virgules (ex: " State-gov" devient "State-gov")
# - na.strings = "?" : R va automatiquement transformer les "?" en NA (valeurs manquantes)
adult_data <- read.csv("adult.data", 
                       header = FALSE, 
                       col.names = noms_colonnes, 
                       strip.white = TRUE, 
                       na.strings = "?",
                       stringsAsFactors = FALSE)

# Ouverture de l'explorateur interactif pour les 1000 premières lignes
View(head(adult_data, 1000))