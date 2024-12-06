from sklearn.preprocessing import normalize
import numpy as np


def pagerank_algorithm(matrix, alpha=0.85, max_iter=100, tol=1e-6):
    """
    Calcule les scores de PageRank à partir d'une matrice de co-occurrence normalisée.

    Cette fonction implémente l'algorithme de PageRank, qui permet d'attribuer un score à chaque élément (ou mot) 
    d'un réseau basé sur les liens ou les co-occurrences, selon l'importance relative de chaque élément.

    Le calcul s'effectue par itérations successives, en mettant à jour les scores à chaque étape, jusqu'à ce qu'une 
    condition de convergence soit atteinte (tolérance `tol`) ou qu'un nombre maximal d'itérations soit effectué.

    Paramètres :
    -----------
    matrix : numpy.ndarray
        La matrice de co-occurrence normalisée dans laquelle les éléments sont liés. 
        La matrice doit être de taille (N, N), où N est le nombre d'éléments (mots, pages, etc.).

    alpha : float, optionnel, par défaut 0.85
        Le facteur d'amortissement (damping factor), qui détermine l'importance de l'exploration des voisins
        par rapport à la possibilité de sauter de manière aléatoire à tout autre élément. 
        Une valeur commune est 0.85.

    max_iter : int, optionnel, par défaut 100
        Le nombre maximal d'itérations pour calculer le PageRank. Si la convergence n'est pas atteinte avant 
        ce nombre d'itérations, le calcul s'arrête.

    tol : float, optionnel, par défaut 1e-6
        La tolérance de convergence. L'algorithme arrête les itérations lorsque la différence L1 entre les 
        nouveaux scores de PageRank et les précédents est inférieure à cette valeur.

    Retourne :
    ---------
    numpy.ndarray
        Un tableau 1D contenant les scores de PageRank pour chaque élément. La somme des scores est égale à 1.
    """
    # Normaliser par norme max, comme pour hits
    matrix_normalized = normalize(matrix, norm='max', axis=1)

    # Initialisation
    N = matrix_normalized.shape[0]
    pagerank = np.ones(N) / N

    # Itérations
    for _ in range(max_iter):
        new_pagerank = alpha * \
            np.dot(matrix_normalized, pagerank) + (1 - alpha) / N

        if np.linalg.norm(new_pagerank - pagerank, 1) < tol:
            break
        pagerank = new_pagerank
    return pagerank
