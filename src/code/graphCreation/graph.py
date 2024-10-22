import glob
from tqdm import tqdm
from dotenv import dotenv_values
import numpy as np
import os
current_working_directory = os.getcwd()


def adjacency_matrix(text, max_distance=1):

    words = text.lower().split(' ')

    unique_words = list(set(words))
    word_index = {word: idx for idx, word in enumerate(unique_words)}

    n = len(unique_words)
    adjacency_matrix = np.zeros((n, n), dtype='uint16')

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


def get_authority_ranking(filepath):

    with open(filepath, encoding="utf8") as f:
        text = f.read()

    matrix, word_index = adjacency_matrix(
        text, max_distance=2)

    #  Reverse or not reverse ?
    reverse_matrix = np.reciprocal(matrix, where=matrix != 0)

    authorities, hubs = hits_algorithm(matrix)

    index_word = {i: word for word, i in word_index.items()}

    sorted_authorities = sorted([(index_word[i], auth) for i, auth in enumerate(
        authorities)], key=lambda x: x[1], reverse=True)

    sorted_hubs = sorted([(index_word[i], hub) for i, hub in enumerate(
        hubs)], key=lambda x: x[1], reverse=True)

    return sorted_authorities, sorted_hubs


authority_ranking_dict = {}
authority_hub_dict = {}

for i, filename in enumerate(tqdm(glob.glob(current_working_directory +
                                            dotenv_values(".env")['PREPROCESSED_FOLDER'] + '*.txt'))):

    sorted_authorities, sorted_hubs = get_authority_ranking(filename)

    authority_ranking_dict[i] = sorted_authorities
    authority_hub_dict[i] = sorted_hubs

print(authority_ranking_dict[0])
