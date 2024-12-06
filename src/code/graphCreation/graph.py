import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
current_working_directory = os.getcwd()


def adjacency_matrix(text, max_distance=1):
    """
    Constructs an adjacency matrix from a given text, representing word co-occurrences.

    This function creates an adjacency matrix where each cell (i, j) represents
    the number of times word i appears within 'max_distance' words of word j in the text.
    The matrix is symmetric, representing an undirected graph.

    Parameters:
    text (str): The input text to analyze.
    max_distance (int, optional): The maximum distance between words to consider
                                  as adjacent. Defaults to 1.

    Returns:
    tuple: A tuple containing two elements:
           - adjacency_matrix (numpy.ndarray): The constructed adjacency matrix.
           - word_index (dict): A dictionary mapping words to their indices in the matrix.

    """
    words = text.lower().split(' ')

    unique_words = list(set(words))
    word_index = {word: idx for idx, word in enumerate(unique_words)}

    n = len(unique_words)
    adjacency_matrix = np.zeros((n, n), dtype='int16')

    for i in range(len(words)):
        word1 = words[i]
        idx1 = word_index[word1]

        for j in range(1, max_distance + 1):
            if i + j < len(words):
                word2 = words[i + j]
                idx2 = word_index[word2]
                adjacency_matrix[idx1, idx2] += 1
                adjacency_matrix[idx2, idx1] += 1  # Graphe non-orienté

    return adjacency_matrix, word_index


def hits_algorithm(adj_matrix, max_iter=100, tol=1e-6):
    """
    Implémentation de l'algorithme HITS pour une matrice d'adjacence.

    :param adj_matrix: Matrice d'adjacence (numpy array)
    :param max_iter: Nombre maximum d'itérations
    :param tol: Tolérance pour la convergence
    :return: Vecteurs d'autorité et de hub
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

    return authorities, hubs


def get_authority_ranking(filepath, k_distance=5):
    """Return a dictionnary of the words in a text ranked according to their PageRank authority"""
    with open(filepath, encoding="utf8") as f:
        text = f.read()

    matrix, word_index = adjacency_matrix(
        text, max_distance=k_distance)

    reciprocal_matrix = np.where(matrix != 0, 1 / matrix, 1e10)

    authorities, hubs = hits_algorithm(reciprocal_matrix)

    index_word = {i: word for word, i in word_index.items()}

    sorted_authorities = sorted([(index_word[i], auth) for i, auth in enumerate(
        authorities)], key=lambda x: x[1], reverse=True)

    sorted_hubs = sorted([(index_word[i], hub) for i, hub in enumerate(
        hubs)], key=lambda x: x[1], reverse=True)

    return sorted_authorities, sorted_hubs


def drawGraph(filepath, k_distance=5):
    """
    Draw the relation graph from a text file.

    This function reads a text file, creates an adjacency matrix representing word co-occurrences,
    and then draws a weighted graph based on this matrix.

    Parameters:
    filepath (str): The path to the text file to be analyzed.
    k_distance (int, optional): The maximum distance between words to consider as adjacent
                                when creating the adjacency matrix. Defaults to 5.

    Returns:
    None: This function does not return anything. It displays the graph using matplotlib.

    The function performs the following steps:
    1. Reads the text file.
    2. Creates an adjacency matrix using the adjacency_matrix function.
    3. Creates a weighted graph from the adjacency matrix.
    4. Draws the graph with labeled nodes and weighted edges.
    5. Displays the graph using matplotlib.
    """
    with open(filepath, encoding="utf8") as f:
        text = f.read()

    matrix, word_index = adjacency_matrix(
        text, max_distance=k_distance)

    node_labels = {i: word for word, i in word_index.items()}

    G_weighted = nx.from_numpy_array(matrix)

    # Dessiner le graphe pondéré avec les poids des arêtes
    pos = nx.spring_layout(G_weighted)
    nx.draw(G_weighted, pos, labels=node_labels,
            node_color='lightgreen', node_size=500, font_size=10)
    edge_labels = nx.get_edge_attributes(G_weighted, 'weight')

    nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels=edge_labels)

    plt.title("Graphe pondéré à partir d'une matrice de poids")
    plt.show()
