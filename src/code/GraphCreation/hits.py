import numpy as np


def hits_algorithm(adj_matrix, max_iter=10, tol=1e-6):
    """
    Calcule le rang des mots dans un graphe en utilisant le sous-graphe k-core.

    Cette fonction extrait le sous-graphe k-core du graphe donné, puis calcule le degré de chaque nœud dans 
    ce sous-graphe pour déterminer le rang des mots. Les mots sont classés en fonction de leur degré (le nombre 
    de connexions qu'ils ont dans le sous-graphe k-core), les mots ayant le plus grand degré étant considérés 
    comme les plus importants.

    Paramètres :
    -----------
    graph : networkx.Graph
        Le graphe à partir duquel le sous-graphe k-core sera extrait.

    k : int, optionnel, par défaut 3
        Le degré minimal des nœuds à conserver dans le sous-graphe k-core. Seuls les nœuds ayant au moins `k` voisins 
        seront inclus dans le calcul des rangs.

    Retourne :
    ---------
    list
        Une liste de tuples (mot, degré), triée par ordre décroissant du degré. Chaque tuple représente un mot du 
        vocabulaire et son degré (son nombre de connexions) dans le sous-graphe k-core.
    """
    n = adj_matrix.shape[0]

    # Initialisation
    hubs = np.ones(n)
    authorities = np.ones(n)

    for _ in range(max_iter):

        # Itérations
        new_authorities = np.dot(adj_matrix.T, hubs)
        new_hubs = np.dot(adj_matrix, new_authorities)

        # Normalisation
        # np.linalg.norm( array, 2) : Largest singular value of array
        new_authorities /= np.linalg.norm(new_authorities, 2)
        new_hubs /= np.linalg.norm(new_hubs, 2)

        if np.linalg.norm(new_authorities - authorities, 2) < tol and np.linalg.norm(new_hubs - hubs, 2) < tol:
            break

        authorities = new_authorities
        hubs = new_hubs

    return authorities
