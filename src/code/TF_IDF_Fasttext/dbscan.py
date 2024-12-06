import numpy as np
import warnings
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import _check_sample_weight, _deprecate_positional_args
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
from collections import defaultdict
from nltk.corpus import stopwords
import re
import spacy
import textacy
from joblib import Parallel, delayed
from scipy.spatial.distance import cosine



@_deprecate_positional_args
def dbscan(X, eps=0.5, *, min_samples=5, metric='minkowski',
           metric_params=None, algorithm='auto', leaf_size=30, p=2,
           sample_weight=None, n_jobs=None, model=None):
    est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm,
                 leaf_size=leaf_size, p=p, n_jobs=n_jobs, model=model)
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class DBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None, allwords=None, model=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000
        self.p = p
        self.stop_words = set(stopwords.words('english'))
        self.n_jobs = n_jobs
        self.clusters = []
        self.model = model
        self.allwords = np.array(
            allwords) if allwords is not None else np.array([])
        self.cluster_dict = {}
        self.cluster_id_counter = 0  # Compteur pour les identifiants de cluster

    def process_corpus(self, documents, n_grams=2, min_n_gram_freq=2):
        """
        Process a corpus of documents to extract tokens and calculate TF-IDF scores.

        Parameters:
        documents (list of str): A list of documents to process.
        n_grams (int, optional): The number of words in each n-gram. Default is 2.
        min_n_gram_freq (int, optional): The minimum frequency for an n-gram to be included. Default is 2.

        Returns:
        all_tokens (set of str): A set of all unique tokens extracted from the documents.
        tf_idf_dict (list of dict): A list of dictionaries, where each dictionary represents the TF-IDF scores for a document.
        """
        all_tokens = set()
        document_count = len(documents)
        tf_idf_dict = [{} for _ in range(len(documents))]
        token_doc_count = defaultdict(int)
        doc = self.nlp("\n".join(documents))
        ngrams = list(textacy.extract.basics.ngrams(
            doc, n_grams, min_freq=min_n_gram_freq))

        for i, full_content in enumerate(documents):
            content = re.sub(
                r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)', "", full_content)
            target_pos = {"VERB", "ADJ", "NUM", "ADP", "PROPN", "PRON", "PUNCT",
                          "SCONJ", "SYM", "ADV", "SPACE", "AUX", "CONJ", "SYM", "PUNCT"}
            target_tags = {"VB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                           "CD", "PRP", "PRP$", "DT", "IN", "CC", "UH", "SYM", "."}
            tokens = {token.lemma_ for token in doc if len(
                token.text) > 3 and token.text not in self.stop_words and token.pos_ not in target_pos and token.tag_ not in target_tags}
            n_grams_doc = {str(n_gram_doc)
                           for n_gram_doc in ngrams if str(n_gram_doc) in content}
            tokens.update(n_grams_doc)
            all_tokens.update(tokens)
            number_content_words = len(full_content.split()) + len(n_grams_doc)
            for token in tokens:
                token_doc_count[token] += 1
                tf_idf_dict[i][token] = (full_content.count(
                    token) / number_content_words if number_content_words > 1 else 0)

        for i in range(document_count):
            for token in tf_idf_dict[i]:
                tf_idf_dict[i][token] *= np.log(document_count / (
                    token_doc_count[token])) if token_doc_count[token] > 1 else 1
        return all_tokens, tf_idf_dict

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.labels_

    def get_words(self, rarray, pos):
        words = []
        for e in rarray:
            self.allwords[e]["cluster_id"] = pos
            self.allwords[e]["in_cluster"] = True
            words.append(self.allwords[e]["word"])
        return words

    def get_clusters(self, tokens, tf_idf_dict):
        self.clusters = []
        all_words = [{"word": token, "in_cluster": False,
                      "cluster_id": None} for token in tokens]
        self.all_words = all_words
        for word in tokens:
            self.add_word(word)
        tf_idf_clusters = self.build_tf_idf_vector_of_cluster(
            self.clusters, tf_idf_dict)
        return self.clusters, tf_idf_clusters

    def get_resumes_doc(self, top_n, n_clusters=4, tokens=None, tf_idf_dict=None):
        clusters, tf_idf_clusters = self.get_clusters(
            tokens=tokens, tf_idf_dict=tf_idf_dict)

        # Calcul des scores des clusters et sélection des meilleurs
        cluster_scores = tf_idf_clusters.sum(axis=0)
        best_cluster_indices = np.argsort(cluster_scores)[
            ::-1][:n_clusters]  # Meilleurs clusters

        # Scores des documents basés sur les meilleurs clusters
        document_scores = tf_idf_clusters[:, best_cluster_indices].sum(axis=1)
        best_document_indices = np.argsort(document_scores)[
            ::-1][:top_n]

        return best_document_indices

    def build_clusters(self):
        for pos in set(self.labels_):
            pos_list = np.where(self.labels_ == pos)[0]
            if pos != -1:
                ref_word = self.allwords[pos_list[0]]["word"]
                cluster_id = self.cluster_id_counter
                self.cluster_id_counter += 1

                self.get_words(pos_list.tolist(), cluster_id)
                centroid_vector = np.mean(
                    [self.model.get_word_vector(ref_word)], axis=0)
                cluster = {
                    'id': cluster_id,
                    'centroid': centroid_vector,
                    'words': [ref_word],
                }
                self.clusters.append(cluster)
                self.cluster_dict[cluster_id] = cluster
            else:
                for e in pos_list:
                    word = self.allwords[e]["word"]
                    cluster_id = self.cluster_id_counter
                    self.cluster_id_counter += 1
                    self.allwords[e]["cluster_id"] = cluster_id
                    self.allwords[e]["in_cluster"] = False
                    cluster = {
                        'id': cluster_id,
                        'centroid': self.model.get_word_vector(word),
                        'words': [word],
                    }
                    self.clusters.append(cluster)
                    self.cluster_dict[cluster_id] = cluster

    def add_word(self, word):
        point = {
            "word": word,
            "in_cluster": False,
            "cluster_id": None,
        }
        n_neighbors = self.epsilon_neighbors(point)

        if not n_neighbors:
            cluster_id = self.cluster_id_counter
            self.cluster_id_counter += 1
            point["cluster_id"] = cluster_id
            self.allwords = np.append(self.allwords, point)
            cluster = {
                'id': cluster_id,
                'centroid': self.model.get_word_vector(word),
                'words': [word],
            }
            self.clusters.append(cluster)
            self.cluster_dict[cluster_id] = cluster
        else:
            self._handle_neighbors(n_neighbors, point, word)

    def _handle_neighbors(self, n_neighbors, point, word):
        on_include = any(neighbor["in_cluster"] for neighbor in n_neighbors)

        if not on_include:
            words = [neighbor["word"] for neighbor in n_neighbors]
            cluster_id = self.cluster_id_counter
            self.cluster_id_counter += 1
            for neighbor in n_neighbors:
                neighbor_idx = np.where(self.allwords == neighbor)[0][0]
                self.allwords[neighbor_idx]["cluster_id"] = cluster_id
                self.allwords[neighbor_idx]["in_cluster"] = True

            words.append(word)
            point["cluster_id"] = cluster_id
            point["in_cluster"] = True
            self.allwords = np.append(self.allwords, point)

            # Paralléliser le calcul des centroids
            centroids = Parallel(n_jobs=self.n_jobs)(
                delayed(self.model.get_word_vector)(w) for w in words)
            centroid_vector = np.mean(centroids, axis=0)

            cluster = {
                'id': cluster_id,
                'centroid': centroid_vector,
                'words': words,
            }
            self.clusters.append(cluster)
            self.cluster_dict[cluster_id] = cluster
        else:
            idhas = n_neighbors[0]["cluster_id"]
            words = self._merge_clusters(n_neighbors, idhas)
            words.append(word)
            point["cluster_id"] = idhas
            point["in_cluster"] = True
            self.allwords = np.append(self.allwords, point)

            # Paralléliser le calcul des centroids
            centroids = Parallel(n_jobs=self.n_jobs)(
                delayed(self.model.get_word_vector)(w) for w in words)
            centroid_vector = np.mean(centroids, axis=0)

            cluster = {
                'id': idhas,
                'centroid': centroid_vector,
                'words': words,
            }
            self.clusters.append(cluster)
            self.cluster_dict[idhas] = cluster

    def _merge_clusters(self, n_neighbors, idhas):
        words = []
        for neighbor in n_neighbors:
            cluster = self.get_cluster(neighbor["cluster_id"])
            if cluster != -1:
                words.extend(cluster["words"])
                if cluster["id"] in self.cluster_dict:
                    del self.cluster_dict[cluster["id"]]
                    self.delete_cluster(cluster["id"])
        self.update_id(words, idhas)
        return words

    def delete_cluster(self, cluster_id):
        self.clusters = [
            cluster for cluster in self.clusters if cluster['id'] != cluster_id]

    def get_cluster(self, cluster_id):
        return self.cluster_dict.get(cluster_id, -1)

    def update_id(self, words, id):
        for word_dict in self.allwords:
            if word_dict["word"] in words:
                word_dict["cluster_id"] = id
                word_dict["in_cluster"] = True

    def epsilon_neighbors(self, P):
        distances = [cosine(self.model.get_word_vector(
            e["word"]), self.model.get_word_vector(P["word"])) for e in self.allwords]
        return [self.allwords[i] for i in range(len(self.allwords)) if distances[i] < self.eps and self.allwords[i]["word"] != P["word"]]

    def build_tf_idf_vector_of_cluster(self, clusters, tf_idf_dict):
        tf_idf_vector = np.zeros((len(tf_idf_dict), len(clusters)))
        cluster_word_sets = {j: set(cluster['words'])
                             for j, cluster in enumerate(clusters)}

        for i, tf_idf in enumerate(tf_idf_dict):
            for k, value in tf_idf.items():
                for j, word_set in cluster_word_sets.items():
                    if k in word_set:
                        tf_idf_vector[i][j] += value * len(word_set)

        return tf_idf_vector
