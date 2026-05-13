#include <iostream> // pour les cout et endl
#include <fstream> // lire les données avec ifstream et écrire les résultats avec ofstream
#include <vector> // STL pour les vecteurs
#include <string> // STL pour les chaines de caractère
#include <sstream> // String Stream pour découper et isoler les variables
#include <cmath> // écriture de l'exponentielle
#include <iomanip> // formate le texte affiché dans la console (nombre de décimales)
#include <Eigen/Dense> // MatrixXd et VectorXd pour représenter les données, résolution de systèmes linéaires et QR
#include <chrono> // pour chronométrer le temps d'exécution

// bibliothèques spécifiques pour l'algorithme Batch
#include <numeric>   // pour std::iota
#include <algorithm> // pour std::shuffle
#include <random>    // pour std::mt19937

using namespace std;
using namespace Eigen;

// =====================================================================================
// FONCTIONS DE BASE ET CHARGEMENT DES DONNÉES
// =====================================================================================

// Hyperparamètre de régularisation Ridge (cf. Section 1.4 du mémoire)
const double LAMBDA_RIDGE = 1e-5; // assez petit pour forcer la Hessienne inversible sans fausser les coefficients

VectorXd sigmoid(const VectorXd& z) { // l'entrée z est le prédicteur linéaire eta = X * beta (voir partie 1.2)
    return (1.0 + (-z).array().exp()).inverse().matrix(); // fonction pi(z) équation 1.2 du mémoire, vectorisée
}

VectorXd get_ridge_penalty(const VectorXd& beta) { // isole la pénalité pour le calcul du gradient
    VectorXd penalty = beta; // copie le vecteur beta
    penalty(0) = 0.0; // Pas de pénalité sur l'Intercept (beta_0) pour lui laisser modéliser la proportion de base
    return penalty; // retourne le vecteur prêt à être multiplié par lambda
}

double compute_cost(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta) { // Calcule la fonction de coût globale J(beta)
    VectorXd P = sigmoid(X * beta); // calcul des probabilités actuelles
    
    // Protection contre le log(0) pour la stabilité numérique
    P = P.cwiseMax(1e-15).cwiseMin(1.0 - 1e-15); // borne les probabilités
    
    // Log-Vraisemblance négative (Minimisation)
    double cost = -(Y.array() * P.array().log() + (1.0 - Y.array()) * (1.0 - P.array()).log()).sum(); // coût classique
    
    // Ajout de la pénalité Ridge
    VectorXd penalty = get_ridge_penalty(beta); // on récupère le vecteur pénalisé (sans l'intercept)
    double ridge_cost = 0.5 * LAMBDA_RIDGE * penalty.squaredNorm(); // norme L2 au carré
    
    return cost + ridge_cost; // retourne le coût total J(beta)
}

VectorXd compute_gradient(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta) { // calcul du gradient analytique
    VectorXd P = sigmoid(X * beta); // calcul des probabilités
    // Gradient de J(beta) : X^T(p - y). Le signe positif ici correspond bien à la minimisation du coût
    VectorXd grad = X.transpose() * (P - Y); // gradient de la logistique
    grad += LAMBDA_RIDGE * get_ridge_penalty(beta); // ajout de la dérivée de la pénalité Ridge
    return grad; // retourne le vecteur gradient
}

// Recherche linéaire d'Armijo (Détaillée à la Section 2.1.2)
// Garantit la convergence
// Ajout d'un paramètre initial_alpha (par défaut à 1.0)
double line_search_armijo(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta, const VectorXd& direction, double initial_alpha = 1.0) {
    double alpha = initial_alpha; // On démarre au pas fourni
    double c = 1e-4; 
    double tau = 0.5; 
    
    double current_cost = compute_cost(X, Y, beta);
    VectorXd grad = compute_gradient(X, Y, beta);
    double m = grad.dot(direction); 

    while (compute_cost(X, Y, beta + alpha * direction) > current_cost + c * alpha * m) { // condition d'Armijo
        alpha *= tau;
        if (alpha < 1e-8) break; 
    }
    return alpha;
}

// =====================================================================================
// ALGORITHMES D'OPTIMISATION (CHAPITRE 2)
// =====================================================================================

// Initialisation intelligente : Log-Odds sur l'Intercept 
VectorXd initialize_beta(int p, const VectorXd& Y) {
    VectorXd beta = VectorXd::Zero(p); // On met tout à zéro
    
    double y_mean = Y.mean(); // Calcule la proportion de Y=1 
    // Sécurité pour éviter log(0) si le dataset est vide ou parfait
    y_mean = std::max(1e-15, std::min(1.0 - 1e-15, y_mean)); 
    
    beta(0) = log(y_mean / (1.0 - y_mean)); // application de la formule : beta_0 = ln(y / (1 - y))
    
    return beta;
}


void run_gradient_descent(const MatrixXd& X, const VectorXd& Y, int max_iter, double tol) { // Descente de gradient classique
    auto start = chrono::high_resolution_clock::now();
    int p = X.cols(); // nombre de variables
    VectorXd beta = initialize_beta(p, Y); // Initialisation intelligente Log-Odds
    
    double current_alpha = 0.001; // Le tout premier pas reste très prudent
    for (int iter = 0; iter < max_iter; ++iter) { // boucle principale
        VectorXd grad = compute_gradient(X, Y, beta); // évaluation du gradient
        VectorXd direction = -grad; // Direction de plus profonde descente
        
        if (grad.norm() < tol) { // critère d'arrêt
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> diff = end - start;
            cout << " [Gradient] Convergence atteinte a l'iteration " << iter << " en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // message de succès
            return; // on sort de la fonction
        }
        current_alpha = line_search_armijo(X, Y, beta, direction, current_alpha * 1.5); 
        
        beta = beta + current_alpha * direction;
        if (iter % 1000 == 0) { // S'exécute seulement 1 fois sur 1000
            cout << "Iteration " << iter << " | Cout = " << compute_cost(X, Y, beta) << "\n";
        }
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << " [Gradient] Arret : maximum d'iterations atteint en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // alerte si non-convergence
}

void run_bfgs(const MatrixXd& X, const VectorXd& Y, int max_iter, double tol) { // Méthode de Quasi-Newton (BFGS)
    auto start = chrono::high_resolution_clock::now();
    int p = X.cols(); // nombre de paramètres
    VectorXd beta = VectorXd::Zero(p); // initialisation
    
    // Initialisation de l'approximation de l'inverse de la Hessienne (Notée D^(0) dans la Section 2.3)
    MatrixXd H_inv = MatrixXd::Identity(p, p); // matrice identité au départ
    VectorXd grad = compute_gradient(X, Y, beta); // premier gradient
    
    for (int iter = 0; iter < max_iter; ++iter) { // boucle principale
        if (grad.norm() < tol) { // critère d'arrêt
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> diff = end - start;
            cout << " [BFGS] Convergence atteinte a l'iteration " << iter << " en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // succès
            cout << " [Apercu] 10 premiers coefficients : " << beta.head(10).transpose() << endl;
            return; // on quitte
        }
        
        VectorXd direction = -H_inv * grad; // calcul de la direction de descente
        
        double alpha = line_search_armijo(X, Y, beta, direction); // recherche d'Armijo pour le pas
        
        VectorXd beta_next = beta + alpha * direction; // application du pas
        VectorXd grad_next = compute_gradient(X, Y, beta_next); // nouveau gradient
        
        // Mise à jour BFGS (cf. équations Section 2.3)
        VectorXd s = beta_next - beta; // différence des positions
        VectorXd y = grad_next - grad; // différence des gradients
        double rho = 1.0 / y.dot(s); // scalaire de normalisation
        
        if (rho > 0) { // vérification de la condition de courbure
            MatrixXd I = MatrixXd::Identity(p, p); // matrice identité
            H_inv = (I - rho * s * y.transpose()) * H_inv * (I - rho * y * s.transpose()) + rho * s * s.transpose(); // formule BFGS
        }
        
        beta = beta_next; // on avance
        grad = grad_next; // on met à jour le gradient
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << " [BFGS] Arret : maximum d'iterations atteint en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // limite atteinte
}

void run_irls(const MatrixXd& X, const VectorXd& Y, int max_iter, double tol, const string& out_filepath) { // Newton-Raphson / IRLS
    auto start = chrono::high_resolution_clock::now();
    int p = X.cols(); // dimensions
    VectorXd beta = VectorXd::Zero(p); // initialisation
    
    for (int iter = 0; iter < max_iter; ++iter) { // boucle de Newton
        VectorXd eta = X * beta; // prédicteur linéaire
        VectorXd P = sigmoid(eta); // probabilités
        P = P.cwiseMax(1e-15).cwiseMin(1.0 - 1e-15); // stabilité
        
        VectorXd W_diag = P.array() * (1.0 - P.array()); // variance locale
        
        // On utilise directement W_diag.asDiagonal() pour que Eigen fasse le produit matriciel virtuellement
        // Hessienne de la fonction de coût J(beta) : X^T W X (Matrice définie positive)
        MatrixXd hessian = X.transpose() * W_diag.asDiagonal() * X; // calcul de courbure
        
        // Ajout de la pénalité Ridge sur la Hessienne (sauf Intercept)
        MatrixXd I = MatrixXd::Identity(p, p); // identité
        I(0, 0) = 0.0; // on ne touche pas à l'intercept
        hessian += LAMBDA_RIDGE * I; // on rend la Hessienne inversible
        
        // Construction de la variable de travail ajustée Z
        VectorXd Z = eta + (Y - P).cwiseQuotient(W_diag); // équation 2.15 du mémoire
        
        // CORRECTION OPTIMISATION : Multiplier une diagonale W par un vecteur Z est un simple produit élément par élément
        VectorXd W_Z = W_diag.cwiseProduct(Z);
        
        // Résolution par décomposition QR pour la stabilité (évite l'inversion directe)
        VectorXd beta_cible = hessian.colPivHouseholderQr().solve(X.transpose() * W_Z); // cible Newton pur
        VectorXd direction = beta_cible - beta; // vecteur de mise à jour
        
        if (direction.norm() < tol) { // si on ne bouge presque plus
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> diff = end - start;
            cout << " [IRLS] Convergence atteinte a l'iteration " << iter << " en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << " (Exportation...)" << endl; // on a gagné
            cout << " [Apercu] 10 premiers coefficients : " << beta.head(10).transpose() << endl;
            // bloc d'exportation vers le CSV
            ofstream out_file(out_filepath); // ouverture fichier
            out_file << "Coefficient\n"; // en-tête
            for(int i = 0; i < p; ++i) { // parcours des betas
                out_file << setprecision(10) << beta(i) << "\n"; // écriture haute précision
            }
            out_file.close(); // fermeture du fichier
            return; // fin
        }
        
        double alpha = line_search_armijo(X, Y, beta, direction); // pas d'Armijo
        beta = beta + alpha * direction; // mise à jour de beta selon le pas amorti 
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cout << " [IRLS] Arret : maximum d'iterations atteint en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // limite
}

void run_minibatch_gradient_descent(const MatrixXd& X, const VectorXd& Y, int max_iter, double learning_rate, int batch_size) { // fonction de descente de gradient par mini-lots
    auto start = chrono::high_resolution_clock::now(); // début du chronométrage
    int n = X.rows(); // récupération du nombre total d'observations
    int p = X.cols(); // récupération du nombre de variables

    VectorXd beta = VectorXd::Zero(p); // initialisation du vecteur des paramètres à zéro

    vector<int> indices(n); // création d'un vecteur pour stocker la position des lignes
    iota(indices.begin(), indices.end(), 0); // remplissage du vecteur avec les entiers de 0 à n-1
    random_device rd; // génération de hasard à partir des composants de l'ordinateur (température processeur, temps de tape sur clavier etc)
    mt19937 g(rd()); // création du générateur pseudo aléatoire (Mersenne Twister)

    for (int iter = 0; iter < max_iter; ++iter) { // boucle principale représentant les époques
        shuffle(indices.begin(), indices.end(), g); // mélange aléatoire des indices pour cette époque
        
        double current_alpha = learning_rate / (1.0 + 0.005 * iter); // décroissance progressive du pas d'apprentissage

        for (int i = 0; i < n; i += batch_size) { // boucle parcourant les données par sauts de la taille du lot
            int current_batch_size = min(batch_size, n - i); // ajustement de la taille si le dernier lot est incomplet

            MatrixXd X_batch(current_batch_size, p); // allocation de la matrice de données pour le lot courant
            VectorXd Y_batch(current_batch_size); // allocation du vecteur cible pour le lot courant
            for (int j = 0; j < current_batch_size; ++j) { // itération pour remplir manuellement le lot
                int idx = indices[i + j]; // récupération de l'indice de la donnée mélangée
                X_batch.row(j) = X.row(idx); // copie de la ligne de caractéristiques correspondante
                Y_batch(j) = Y(idx); // copie de la variable à prédire correspondante
            }

            VectorXd eta = X_batch * beta; // calcul du prédicteur linéaire sur le lot
            VectorXd pi = sigmoid(eta); // calcul des probabilités prédites via la fonction d'activation

            VectorXd grad = (X_batch.transpose() * (pi - Y_batch)) / current_batch_size; // calcul du gradient moyen du lot
            grad += (2.0 * LAMBDA_RIDGE / n) * get_ridge_penalty(beta); // ajout de la dérivée de la pénalité ridge

            beta -= current_alpha * grad; // mise à jour des paramètres dans le sens inverse du gradient
        }

        if (iter % 1000 == 0) { // condition pour ne pas inonder la console
            cout << "Iteration " << iter << " | Cout = " << compute_cost(X, Y, beta) << "\n"; // calcul et affichage du coût global sur toutes les données
        }
    }

    auto end = chrono::high_resolution_clock::now(); // fin du chronométrage
    chrono::duration<double> diff = end - start; // calcul du temps écoulé
    cout << " [Mini-Batch] Arret : maximum d'iterations atteint en " << diff.count() << " s | Cout final : " << compute_cost(X, Y, beta) << endl; // affichage du bilan de fin de fonction
}

// =====================================================================================
// LECTURE CSV ET PROGRAMME PRINCIPAL
// =====================================================================================

void load_csv(const string& filepath, MatrixXd& X, VectorXd& Y) { 
    ifstream file(filepath); // ouvre le CSV
    string line; // ligne courante
    vector<vector<double>> data; // stockage dynamique
    
    while (getline(file, line)) { // lit ligne par ligne
        stringstream ss(line); // convertit en flux
        string val; // valeur texte
        vector<double> row; // ligne numérique
        while (getline(ss, val, ',')) { // découpe aux virgules
            row.push_back(stod(val)); // convertit en double
        }
        data.push_back(row); // ajoute la ligne
    }
    
    int n = data.size(); // nombre de lignes
    int p = data[0].size() - 1; // nombre de variables X (on enlève Y)
    
    X.resize(n, p); // on dimensionne Eigen
    Y.resize(n); // on dimensionne Eigen
    
    for (int i = 0; i < n; ++i) { // on remplit les matrices
        Y(i) = data[i][0]; // la première colonne est la cible
        for (int j = 0; j < p; ++j) { // le reste
            X(i, j) = data[i][j + 1]; // matrice de design
        }
    }
}

void process_dataset(const string& name, const string& filepath, const string& out_filepath) { // Moteur d'exécution
    MatrixXd X; // création d'une matrice vide
    VectorXd Y; // création d'un vecteur vide
    load_csv(filepath, X, Y); // chargement du CSV et remplissage
    
    cout << "\n=========================================================" << endl;
    cout << " TRAITEMENT : " << name << endl;
    cout << " Dimensions : n = " << X.rows() << " observations | p+1 = " << X.cols() << " variables" << endl;
    cout << "=========================================================" << endl;
    
    // lancement des algorithmes pour le benchmark de vitesse
    run_gradient_descent(X, Y, 2000, 1e-4); // lent (40 secondes)
    run_minibatch_gradient_descent(X, Y, 2000, 0.5, 128); // lent (30 secondes)
    run_bfgs(X, Y, 500, 1e-4); // rapide
    
    
    // L'IRLS est lancé en dernier pour exporter les coefficients finaux exacts 
    run_irls(X, Y, 100, 1e-4, out_filepath); // très rapide
}

int main() { // point d'entrée
    // On ne traite que le jeu d'entrainement pour générer les coefficients. 
    // Le jeu de test sera géré en R pour les prédictions.
    process_dataset("Modele Adult (Train)", "export_adult_train.csv", "beta_adult_train.csv"); // exécution
    return 0; // fin de programme
}