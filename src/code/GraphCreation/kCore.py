import networkx as nx


def create_graph_from_matrix(matrix, vocab, threshold=1):
    """
    Crée un graphe à partir d'une matrice de co-occurrence et d'un vocabulaire.

    Cette fonction génère un graphe non orienté où chaque nœud représente un mot du vocabulaire, 
    et les arêtes entre les nœuds sont créées en fonction des valeurs de la matrice de co-occurrence. 
    Une arête est ajoutée entre deux mots si leur valeur de co-occurrence dépasse un seuil donné.

    Paramètres :
    -----------
    matrix : numpy.ndarray
        La matrice de co-occurrence de taille (n, n) représentant les relations de co-occurrence entre 
        les mots dans le vocabulaire. La valeur (i, j) de la matrice indique la co-occurrence entre les 
        mots vocab[i] et vocab[j].

    vocab : list
        Une liste des mots représentant le vocabulaire, avec un ordre qui correspond aux indices de 
        la matrice de co-occurrence. Le vocabulaire contient `n` mots si la matrice est de taille (n, n).

    threshold : float, optionnel, par défaut 1
        Le seuil de co-occurrence minimum nécessaire pour ajouter une arête entre deux mots. Si la valeur
        de co-occurrence entre deux mots est inférieure à ce seuil, aucune arête ne sera ajoutée.

    Retourne :
    ---------
    networkx.Graph
        Un graphe non orienté où chaque nœud est un mot du vocabulaire, et chaque arête représente 
        une co-occurrence entre deux mots dont la valeur dépasse le seuil.
    """
    G = nx.Graph()

    for word in vocab:
        G.add_node(word)

    for i in range(len(vocab)):
        for j in range(i + 1, len(vocab)):
            if matrix[i, j] > threshold:
                G.add_edge(vocab[i], vocab[j], weight=matrix[i, j])

    return G


def k_core(graph, k=3):
    """
    Extrait le sous-graphe k-core d'un graphe donné.

    Le sous-graphe k-core contient tous les nœuds du graphe original qui ont au moins `k` voisins dans le graphe. 
    Les nœuds qui ne satisfont pas cette condition sont supprimés, ainsi que toutes les arêtes associées. 
    Cela permet de réduire le graphe aux nœuds les plus importants et leurs connexions.

    Paramètres :
    -----------
    graph : networkx.Graph
        Le graphe à partir duquel le sous-graphe k-core sera extrait.

    k : int, optionnel, par défaut 3
        Le degré minimal des nœuds à conserver dans le sous-graphe. Seuls les nœuds ayant au moins `k` voisins 
        seront conservés dans le sous-graphe.

    Retourne :
    ---------
    networkx.Graph
        Le sous-graphe k-core du graphe original, contenant uniquement les nœuds avec au moins `k` voisins.
    """
    # Prevents self-loop in graph
    graph.remove_edges_from(nx.selfloop_edges(graph))

    kcore_subgraph = nx.k_core(graph, k)

    return kcore_subgraph


def get_word_ranks(graph, k=3):
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
    kcore_subgraph = k_core(graph, k)

    degrees = dict(kcore_subgraph.degree())

    sorted_words = sorted(
        degrees.items(), key=lambda item: item[1], reverse=True)

    return sorted_words
