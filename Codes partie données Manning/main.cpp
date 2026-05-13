#include <iostream> // pour les cout et endl
#include <fstream> // lire les données avec ifstream et écrire les résultats avec ofstream
#include <vector> // STL pour les vecteurs
#include <string> // STL pour les chaines de caractère
#include <sstream> // String Stream pour découper et isoler les variables
#include <cmath> // écriture de l'exponentielle
#include <iomanip> // formate le texte affiché la console (nombre de décimales)
#include <Eigen/Dense> // MatrixXd et VectorXd pour  pour représenter les données sans allocation de mémoire
                       // mécanique matricielle (transposition, résolution de systèmes linéaires et QR, produits X^t W X)
using namespace std;
using namespace Eigen;

// =====================================================================================
// FONCTIONS DE BASE ET CHARGEMENT DES DONNÉES
// =====================================================================================

// Hyperparamètre de régularisation Ridge (cf. Section 1.4 du mémoire)
// Évite la singularité de la matrice Hessienne en cas de séparation quasi-complète
const double LAMBDA_RIDGE = 1e-5; // assez petit pour forcer la Hessienne inversible mais pas assez grand pour fausser les coefficients

// Fonction de lien logistique standard
VectorXd sigmoid(const VectorXd& z) { // l'éntrée z est le prédicteur linéaire eta = X * beta  (voir partie 1.2)
    return (1.0 + (-z).array().exp()).inverse().matrix(); // fonction pi(z) équation 1.2 du mémoire, vectorisée avec Eigen
} // on traite le vecteur -z comme un tableau, .exp applique l'exponentielle sur chaque ligne, .inverse inverse tous les éléments et .matrix convertit le résultat en vecteur colonne

// Fonction pour isoler la pénalité Ridge
// Fondamental : on ne pénalise jamais la constante (l'Intercept beta_0)
VectorXd get_ridge_penalty(const VectorXd& beta) { // copie le vecteur de paramètres beta et force le premier coefficient à être 0
    VectorXd penalty = beta;
    penalty(0) = 0.0; 
    return penalty;
}

// Chargement des matrices depuis les fichiers CSV préparés par R
void load_csv(const string& filename, MatrixXd& X, VectorXd& Y) {
    ifstream file(filename);
    string line, val;
    vector<vector<double>> data; // tableau dynamique temporaire car on ne connait pas la taille de l'entrée
    
    while (getline(file, line)) { // lit le fichier ligne par ligne
        stringstream ss(line); // transforme la ligne en flux découpable
        vector<double> row; 
        while (getline(ss, val, ',')) { row.push_back(stod(val)); } // découpe le flux à chaque virgule
        // stod(val) transforme de string à double l'élément val
        data.push_back(row);
    }
    
    int n = data.size(); // compte le nombre d'observations n (nombre de ligne du vecteur global)
    if (n == 0) return;
    int p = data[0].size() - 1; // La première colonne est la variable cible Y donc on soustrait 1 pour avoir le nombre de variables p
    X.resize(n, p); // réajuste la taille de l'entrée X maintenant qu'on connait les dimensions
    Y.resize(n); // réajuste la taille de l'entrée Y
    
    for (int i = 0; i < n; ++i) { // on distribue les données
        Y(i) = data[i][0]; // la première colonne est envoyé dans Y 
        for (int j = 0; j < p; ++j) { X(i, j) = data[i][j + 1]; } // les autres colonbnes sont envoyées dans X 
    }
}

// Exportation brute des coefficients beta pour l'analyse ultérieure sous R
void export_betas(const string& export_filename, const VectorXd& beta) {
    ofstream file(export_filename); // ouverture du fichier d'export
    file << "Coefficient\n"; // Première ligne de notre fichier (donne un nom de colonne pour éviter que R considère le premier coefficient comme un nom)
    for(int j = 0; j < beta.size(); ++j) {
        file << setprecision(8) << beta(j) << "\n"; // on force le nombre de décimales à 8
    }
    file.close();
    cout << " -> Coefficients optimaux (beta) exportes dans : " << export_filename << endl; // confirmation d'exportation
}

// =====================================================================================
// FONCTIONS OBJECTIF ET GRADIENT (Équations du Chapitre 1 & 2)
// =====================================================================================

// Calcul de la fonction de coût J(beta) = -Log-Vraisemblance + Pénalité Ridge
double compute_cost(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta) {
    VectorXd P = sigmoid(X * beta); // calcul du vecteur p noté p_i(beta) dans le mémoire
    double cost = 0.0;
    double eps = 1e-15; // garde-fou, limite les probabilités pour éviter log(0) = -infini
    
    for(int i = 0; i < Y.size(); ++i) {
        double p_i = max(min(P(i), 1.0 - eps), eps); // on force à être dans l'intervalle [eps, 1-eps]
        cost -= Y(i) * log(p_i) + (1.0 - Y(i)) * log(1.0 - p_i); // équation de l'opposée de la log vraisemblance 
    }
    VectorXd beta_penalized = get_ridge_penalty(beta);     // ajout de la composante de régularisation L2 (Ridge)
    cost += 0.5 * LAMBDA_RIDGE * beta_penalized.squaredNorm(); // calcul du cout final pénalisé
    
    return cost;
}

// Calcul du Gradient Analytique de la fonction de coût
VectorXd compute_gradient(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta) {
    VectorXd P = sigmoid(X * beta); // on génère les proba prédites dans P
    return X.transpose() * (P - Y) + LAMBDA_RIDGE * get_ridge_penalty(beta); // gradient de J(beta) : X^T(p - y). Le signe correspond à la minimisation du coût.
}

// Recherche linéaire d'Armijo (Détaillée à la Section 2.1.2)
// Garantit la convergence
double line_search_armijo(const MatrixXd& X, const VectorXd& Y, const VectorXd& beta, const VectorXd& direction) {
    double alpha = 1.0; // pas initial
    double c = 1e-4; // constante de pente très faible
    double tau = 0.5; // Procédure de Step-halving
    
    double current_cost = compute_cost(X, Y, beta);
    VectorXd grad = compute_gradient(X, Y, beta);
    double m = grad.dot(direction); // produit scalaire entre le gradient du cout et la direction 

    // Vérification de la condition d'Armijo
    while (compute_cost(X, Y, beta + alpha * direction) > current_cost + c * alpha * m) {
        alpha *= tau;
        if (alpha < 1e-8) break; // Sécurité pour forcer l'arrêt si le pas devient microscopique
    }
    return alpha;
}

// =====================================================================================
// MOTEURS ALGORITHMIQUES D'OPTIMISATION (Chapitre 2)
// =====================================================================================

// 1. DESCENTE DE GRADIENT (Section 2.1)
void run_gradient_descent(const MatrixXd& X, const VectorXd& Y, int max_iter, double epsilon) {
    cout << "  * Algorithme 1 : Descente de Gradient (Initialisation...)" << endl;
    VectorXd beta = VectorXd::Zero(X.cols()); // on part de beta qui est le vecteur nul dans R^p+1 donc des probabilités qui valent 0.5 pour chaque individu
    
    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd grad = compute_gradient(X, Y, beta); // calcul du gradient
        VectorXd direction = -grad; // Direction de plus profonde descente
        
        if (grad.norm() < epsilon) { // tolérance pour l'annulation du gradient
            cout << "    -> Convergence de la Descente de Gradient en " << iter << " iterations. (Cout final: " << compute_cost(X, Y, beta) << ")" << endl;
            return;
        }
        
        double alpha = line_search_armijo(X, Y, beta, direction); // recherche d'Armijo pour le pas
        beta = beta + alpha * direction; // mise à joure de beta selon la formule de Descente de Gradient
    }
    cout << "    -> La Descente de Gradient a atteint la limite d'iterations sans converger." << endl;
}

// 2. QUASI-NEWTON BFGS (Section 2.3)
void run_bfgs(const MatrixXd& X, const VectorXd& Y, int max_iter, double epsilon) {
    cout << "  * Algorithme 2 : Quasi-Newton BFGS (Initialisation...)" << endl;
    int p = X.cols();
    VectorXd beta = VectorXd::Zero(p); // on part du vecteur nul
    
    // Initialisation de l'approximation de l'inverse de la Hessienne (Notée D^(0) dans la Section 2.3)
    MatrixXd H_inv = MatrixXd::Identity(p, p); // on part de l'identité comme approximation de H
    MatrixXd I = MatrixXd::Identity(p, p);
    
    VectorXd grad = compute_gradient(X, Y, beta); // calcul du gradient

    for (int iter = 0; iter < max_iter; ++iter) {
        if (grad.norm() < epsilon) { // tolérance d'annulation du gradient
            cout << "    -> Convergence BFGS en " << iter << " iterations. (Cout final: " << compute_cost(X, Y, beta) << ")" << endl;
            return;
        }
        
        VectorXd direction = -H_inv * grad; //  on multiplie l'opposée de la Hessienne approximée par notre gradient pour tenir compte de la courbure
        double alpha = line_search_armijo(X, Y, beta, direction); // recherche d'Armijo pour le pas
        
        VectorXd beta_new = beta + alpha * direction; // mise à jour de beta
        VectorXd grad_new = compute_gradient(X, Y, beta_new); // calcul du nouveau gradient en le nouveau beta
        
        // Mise à jour de H_inv via la formule de Sherman-Morrison (BFGS Update)
        VectorXd s = beta_new - beta; // variation de la position
        VectorXd y = grad_new - grad; // variation du gradient
        
        double ys = y.dot(s);   // produit entre y et s
        if (ys > 1e-10) {  // vérification de la positivité du produit (condition de courbure qui garantie que l'inverse de la Hessienne est définie positive)
            double rho = 1.0 / ys; 
            H_inv = (I - rho * s * y.transpose()) * H_inv * (I - rho * y * s.transpose()) + rho * s * s.transpose(); //mise à jour de l'inverse de la hessienne approximée
        }
        // mise à jour pour l'itération suivante
        beta = beta_new; 
        grad = grad_new;
    }
}

// 3. IRLS AMORTI / DAMPED NEWTON (Section 2.2.5)
// c'est cet algorithme qui génère les coefficients finaux exportés
void run_irls(const MatrixXd& X, const VectorXd& Y, int max_iter, double epsilon, const string& out_file) {// out_file pour exporter les coefficients
    cout << "  * Algorithme 3 : IRLS Amorti / Damped Newton (Initialisation...)" << endl;
    int p = X.cols();
    VectorXd beta = VectorXd::Zero(p); // initialisation avec beta en vecteur nul
    MatrixXd hessian; 
    
    MatrixXd I_ridge = MatrixXd::Identity(p, p);  // matrice pour la pénalité Ridge
    I_ridge(0, 0) = 0.0; // Pas de pénalité sur l'Intercept

    for (int iter = 0; iter < max_iter; ++iter) {
        VectorXd eta = X * beta;
        VectorXd P = sigmoid(eta);

        VectorXd W_diag = P.array() * (1.0 - P.array());  // écriture d'un vecteur contenant les w_i
        W_diag = W_diag.cwiseMax(1e-10);  // saturation contrôlée pour W (évite les cas 0 et 1)
        MatrixXd W = W_diag.asDiagonal(); // écriture de la matrice W dont la diagonale est le vecteur W_diag

        VectorXd Z = eta.array() + (Y - P).array() / W_diag.array(); // création de la variable ajustée Z (Section 2.2.2)
        
        hessian = X.transpose() * W * X + LAMBDA_RIDGE * I_ridge; // Hessienne exacte + Régularisation (Section 1.4)
        
        // résolution robuste de H * Beta  = X^t * W * Z par décomposition QR pour trouver beta sans inversion (Section 2.2.4)
        VectorXd beta_cible = hessian.colPivHouseholderQr().solve(X.transpose() * W * Z); 
        
        // calcul de la direction et application du pas amorti (Armijo)
        VectorXd direction = beta_cible - beta; 
        double alpha = line_search_armijo(X, Y, beta, direction); // pas d'Armijo
        
        beta = beta + alpha * direction; // mise à jour de beta selon le pas amorti 

        // condition d'arrêt basée sur la norme de la direction
        if (direction.norm() < epsilon) {
            cout << "    -> Convergence IRLS en " << iter + 1 << " iterations. (Cout final: " << compute_cost(X, Y, beta) << ")" << endl;
            // exportation des coefficients optimaux
            export_betas(out_file, beta); 
            return;
        }
    }
}

// fonction qui assemble tout le travail pour chaque jeu de données
void process_dataset(const string& name, const string& filepath, const string& out_filepath) {
    MatrixXd X; // création d'une matrice vide
    VectorXd Y; // création d'un vecteur vide
    load_csv(filepath, X, Y); // chargement du CSV et remplissage des matrices et vecteurs de données
    
    cout << "\n=========================================================" << endl;
    cout << " TRAITEMENT : " << name << endl;
    cout << " Dimensions : n = " << X.rows() << " observations | p+1 = " << X.cols() << " variables" << endl;
    cout << "=========================================================" << endl;
    
    // lancement des algorithmes
    run_gradient_descent(X, Y, 5000, 1e-4);
    run_bfgs(X, Y, 500, 1e-4);
    // L'IRLS est lancé en dernier pour exporter les coefficients finaux de la méthode de référence
    run_irls(X, Y, 100, 1e-4, out_filepath); 
}

int main() {
    process_dataset("Modele Historique (Manning)", "export_manning.csv", "beta_manning.csv");
    process_dataset("Modele Meilleur Ratio (BIC)", "export_best_ratio.csv", "beta_best_ratio.csv");
    process_dataset("Modele Variables Brutes (Stepwise)", "export_best_brut.csv", "beta_best_brut.csv");
    
    cout << "\n[Termine] Tous les modeles ont ete traites. Retour a R pour l'inference." << endl;
    return 0;
}