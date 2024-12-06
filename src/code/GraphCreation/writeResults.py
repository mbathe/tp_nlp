import glob
from tqdm import tqdm
from dotenv import dotenv_values
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from kCore import create_graph_from_matrix, get_word_ranks
from hits import hits_algorithm
from pagerank import pagerank_algorithm

current_working_directory = os.getcwd()

# TODO : https://scikit-network.readthedocs.io/en/latest/tutorials/ranking/pagerank.html


def adjacency_matrix(text, max_distance=1):
    """
    Crée une matrice d'adjacence basée sur les co-occurrences des mots dans un texte.

    Cette fonction génère une matrice d'adjacence où chaque entrée (i, j) indique le nombre de fois où
    le mot i et le mot j apparaissent à une distance maximale donnée dans le texte. La distance entre
    deux mots est calculée comme la différence d'indices dans le texte, et les arêtes sont ajoutées entre
    deux mots s'ils sont suffisamment proches.

    Paramètres :
    -----------
    text : str
        Le texte d'entrée où les mots seront analysés pour leurs co-occurrences.

    max_distance : int, optionnel, par défaut 1
        La distance maximale entre deux mots pour qu'ils soient considérés comme co-occurrents. Si la distance
        entre deux mots dans le texte est inférieure ou égale à `max_distance`, une arête est ajoutée dans la matrice.

    Retourne :
    ---------
    tuple : (numpy.ndarray, dict)
        - numpy.ndarray : La matrice d'adjacence où chaque entrée (i, j) représente le nombre de co-occurrences 
          entre les mots i et j dans le texte.
        - dict : Un dictionnaire où chaque mot du texte est associé à un indice unique dans la matrice.
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


def get_score_ranking(filepath, k_distance=5, method='hits'):
    """
    Retourne un dictionnaire des mots d'un texte classés selon leur autorité ou leur importance selon une méthode donnée.

    Cette fonction lit le texte à partir d'un fichier, crée une matrice d'adjacence pour les mots, puis utilise
    une des méthodes d'algorithmes ('hits', 'pagerank', 'kcore') pour calculer un score pour chaque mot. Les mots
    sont ensuite triés par score et les 10 mots les plus importants sont retournés.

    Paramètres :
    -----------
    filepath : str
        Le chemin vers le fichier texte à analyser.

    k_distance : int, optionnel, par défaut 5
        La distance maximale entre deux mots pour les considérer comme co-occurrents lors de la construction de la matrice
        d'adjacence.

    method : str, optionnel, par défaut 'hits'
        La méthode à utiliser pour le calcul des scores des mots. Peut être l'une des trois suivantes :
        - 'hits' : Utilise l'algorithme HITS pour calculer l'autorité des mots.
        - 'pagerank' : Utilise l'algorithme PageRank pour calculer l'importance des mots.
        - 'kcore' : Utilise l'algorithme k-core pour extraire les mots les plus connectés dans le graphe.

    Retourne :
    ---------
    list
        Une liste des 10 mots les plus importants dans le texte, triée selon leur score d'autorité ou d'importance,
        selon la méthode choisie.
    """
    with open(filepath, encoding="utf8") as f:
        text = f.read()

    matrix, word_index = adjacency_matrix(
        text, max_distance=k_distance)

    reciprocal_matrix = np.where(matrix > 1e-10, 1 / matrix, 1e10)

    if method == 'hits':

        score = hits_algorithm(reciprocal_matrix)
        index_word = {i: word for word, i in word_index.items()}

        sorted_score = sorted([(index_word[i], auth) for i, auth in enumerate(
            score)], key=lambda x: x[1], reverse=True)

        return (list(reversed([temp[0]
                              for temp in sorted_score[-10:]])))

    elif method == 'pagerank':

        score = pagerank_algorithm(reciprocal_matrix)
        index_word = {i: word for word, i in word_index.items()}

        sorted_score = sorted([(index_word[i], auth) for i, auth in enumerate(
            score)], key=lambda x: x[1], reverse=True)

        return (list(reversed([temp[0]
                               for temp in sorted_score[-10:]])))

    elif method == 'kcore':

        graph = create_graph_from_matrix(
            matrix, list(word_index.keys()))

        word_ranks = get_word_ranks(graph, k=2)

        return ([word for word, rank in word_ranks[:10]])


def drawGraph(filepath, k_distance=5):
    """
    Trace un graphe représentant les relations de co-occurrence des mots dans un texte.

    Cette fonction lit un texte depuis un fichier, génère une matrice d'adjacence des mots et crée un graphe pondéré
    où chaque nœud représente un mot et chaque arête représente une co-occurrence entre deux mots. Le graphe est ensuite
    tracé avec les poids des arêtes indiquant la force de la co-occurrence.

    Paramètres :
    -----------
    filepath : str
        Le chemin vers le fichier texte à analyser.

    k_distance : int, optionnel, par défaut 5
        La distance maximale entre deux mots pour les considérer comme co-occurrents lors de la création du graphe.

    Retourne :
    ---------
    None
        La fonction génère simplement un graphique affiché avec `matplotlib`.
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


def writeResultInExcel():
    """
    Écrit les résultats des analyses de mots importants pour plusieurs textes dans un fichier Excel.

    Cette fonction lit les fichiers texte prétraités dans un répertoire donné, applique les algorithmes HITS, PageRank 
    et k-core pour chaque texte pour déterminer les mots les plus importants, puis enregistre les résultats dans un fichier Excel.
    Les mots les plus importants sont organisés par texte et par algorithme, avec leur rang dans une colonne distincte.

    Paramètres :
    -----------
    Aucun paramètre n'est requis pour cette fonction.

    Retourne :
    ---------
    None
        La fonction n'a pas de valeur de retour. Elle crée et écrit un fichier Excel avec les résultats des analyses.
    """
    results = {}

    for i, filename in enumerate(tqdm(glob.glob(current_working_directory +
                                                dotenv_values(".env")['PREPROCESSED_FOLDER'] + '*.txt'))):
        results[i] = {}

        sorted_words_hits = get_score_ranking(
            filename, k_distance=10, method='hits')

        results[i]['hits'] = sorted_words_hits

        sorted_words_pagerank = get_score_ranking(
            filename, k_distance=10, method='pagerank')

        results[i]['pagerank'] = sorted_words_pagerank

        sorted_words_kcore = get_score_ranking(
            filename, k_distance=10, method='kcore')

        results[i]['kcore'] = sorted_words_kcore

    data = []

    for texte, algos in results.items():
        for algo, words in algos.items():
            for idx, word in enumerate(words):
                data.append([texte, algo, f'Rank {idx+1}', word])

    df = pd.DataFrame(data, columns=['Texte', 'Algorithme', 'Rang', 'Mot'])

    df.to_excel('resultats/mots_importants_graph_txt.xlsx',
                index=False, engine='openpyxl')


writeResultInExcel()
