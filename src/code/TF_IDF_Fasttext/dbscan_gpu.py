import cupy as cp
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import _deprecate_positional_args
from nltk.corpus import stopwords
import spacy
from joblib import Parallel, delayed


from collections import defaultdict
import cupy as cp
from collections import defaultdict
from nltk.corpus import stopwords
import re
import textacy
import spacy
import torch
from torch import amp
import math


class TextPreprocessor:
    def __init__(self, batch_size=32):
        """
        
        Initialize with GPU support
        batch_size: Number of documents to process in parallel on GPU
        
        
        """
        spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.target_pos = {"VERB", "ADJ", "NUM", "ADP", "PROPN", "PRON",
                           "PUNCT", "SCONJ", "SYM", "ADV", "SPACE", "AUX", "CONJ"}
        self.target_tags = {"VB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                            "CD", "PRP", "PRP$", "DT", "IN", "CC", "UH", "SYM", "."}

    def clean_document_batch(self, batch):
        """Nettoie un batch de documents en parallèle sur GPU"""
        pattern = r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)'
        return [re.sub(pattern, "", doc) for doc in batch]

    def process_batch_gpu(self, batch, ngrams):
        """Traite un batch de documents sur GPU"""
        cleaned_batch = self.clean_document_batch(batch)
        docs = list(self.nlp.pipe(cleaned_batch, batch_size=len(batch)))

        results = []
        for i, doc in enumerate(docs):
            with amp.autocast(device_type=self.device.type):
                tokens = {token.lemma_ for token in doc
                          if len(token.text) > 3
                          and token.text not in self.stop_words
                          and token.pos_ not in self.target_pos
                          and token.tag_ not in self.target_tags}

                n_grams_doc = {ng for ng in ngrams if ng in cleaned_batch[i]}
                tokens.update(n_grams_doc)

                number_content_words = len(
                    cleaned_batch[i].split()) + len(n_grams_doc)

                tf_dict = {}
                if number_content_words > 1:
                    doc_tensor = torch.tensor(
                        [ord(c) for c in cleaned_batch[i]], device=self.device)
                    for token in tokens:
                        token_tensor = torch.tensor(
                            [ord(c) for c in token], device=self.device)
                        count = torch.sum(
                            doc_tensor.view(-1, 1) == token_tensor).item()
                        tf_dict[token] = count / number_content_words
                else:
                    tf_dict = {token: 0 for token in tokens}

            results.append((i, tokens, tf_dict))
        return results

    def preprocess(self, documents, n_grams=2, min_n_gram_freq=2):
        """Traitement du corpus avec accélération GPU"""
        document_count = len(documents)

        # Extraction des n-grams sur tout le corpus
        joined_docs = "\n".join(documents)
        doc = self.nlp(joined_docs)
        ngrams = {str(n_gram) for n_gram in textacy.extract.basics.ngrams(
            doc, n_grams, min_freq=min_n_gram_freq)}

        all_tokens = set()
        tf_idf_dict = [{} for _ in range(document_count)]
        token_doc_count = defaultdict(int)

        num_batches = math.ceil(len(documents) / self.batch_size)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            batch_results = self.process_batch_gpu(batch, ngrams)
            for doc_idx, tokens, tf_dict in batch_results:
                actual_idx = start_idx + doc_idx
                all_tokens.update(tokens)
                tf_idf_dict[actual_idx] = tf_dict
                for token in tokens:
                    token_doc_count[token] += 1

        doc_count_tensor = cp.array(document_count)
        for i in range(document_count):
            doc_dict = tf_idf_dict[i]
            if doc_dict:
                token_counts = cp.array([token_doc_count[token]
                                        for token in doc_dict])
                tf_values = cp.array(list(doc_dict.values()))

                idf = cp.log(doc_count_tensor / cp.maximum(token_counts, 1))
                tf_idf = tf_values * idf

                tf_idf_dict[i] = {token: float(
                    score) for token, score in zip(doc_dict.keys(), tf_idf)}

        return all_tokens, tf_idf_dict

    def __del__(self):
        # torch.cuda.empty_cache()
        print()


@_deprecate_positional_args
def dbscan(X, eps=0.5, *, min_samples=5, metric='cosine',
           metric_params=None, algorithm='auto', leaf_size=30, p=2,
           sample_weight=None, n_jobs=None, model=None):
    est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm,
                 leaf_size=leaf_size, p=p, n_jobs=n_jobs, model=model)
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_


class DBSCAN(ClusterMixin, BaseEstimator):
    def __init__(self, eps=0.5, *, min_samples=5, metric='cosine',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None, allwords=None, model=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.nlp = spacy.load("en_core_web_sm")
        self.p = p
        self.stop_words = set(stopwords.words('english'))
        self.n_jobs = n_jobs
        self.clusters = []
        self.model = model
        self.allwords = cp.array(
            allwords) if allwords is not None else cp.array([])
        self.cluster_dict = {}
        self.cluster_id_counter = 0

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.labels_

    def epsilon_neighbors(self, P):
        P_vector = self.model.get_word_vector(P["word"])

        # Obtenir tous les vecteurs et les convertir en tableau CuPy
        all_vectors = cp.array(
            [self.model.get_word_vector(e["word"])
             for e in self.allwords]
        )

        # Normaliser le vecteur P
        P_vector = cp.asarray(P_vector)
        # print(P_vector.shape)
        norm_P_vector = cp.linalg.norm(P_vector)
        if norm_P_vector == 0:
            raise ValueError("Le vecteur P ne doit pas être nul.")

        P_vector_normalized = P_vector / norm_P_vector

        # Normaliser tous les vecteurs
        norms = cp.linalg.norm(all_vectors, axis=1)
        norms[norms == 0] = 1  # Éviter la division par zéro
        all_vectors_normalized = all_vectors / norms[:, cp.newaxis]
        # print(all_vectors_normalized.shape, P_vector_normalized.shape)

        # Calculer les distances cosinus
        distances = 1 - cp.dot(all_vectors_normalized, P_vector_normalized)

        # Retourner les voisins dont la distance est inférieure à eps
        return [self.allwords[i] for i in range(len(self.allwords)) if distances[i] < self.eps]

    def add_word(self, word):
        point = {"word": word, "in_cluster": False, "cluster_id": None}
        n_neighbors = self.epsilon_neighbors(point)

        if not n_neighbors:
            cluster_id = self.cluster_id_counter
            self.cluster_id_counter += 1
            point["cluster_id"] = cluster_id
            self.allwords = cp.append(self.allwords, point)
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
                neighbor_idx = cp.where(self.allwords == neighbor)[0][0]
                self.allwords[neighbor_idx]["cluster_id"] = cluster_id
                self.allwords[neighbor_idx]["in_cluster"] = True

            words.append(word)
            point["cluster_id"] = cluster_id
            point["in_cluster"] = True
            self.allwords = cp.append(self.allwords, point)

            centroids = Parallel(n_jobs=self.n_jobs)(
                delayed(self.model.get_word_vector)(w) for w in words)
            centroid_vector = cp.mean(centroids, axis=0)

            cluster = {'id': cluster_id,
                       'centroid': centroid_vector, 'words': words}
            self.clusters.append(cluster)
            self.cluster_dict[cluster_id] = cluster
        else:
            idhas = n_neighbors[0]["cluster_id"]
            words = self._merge_clusters(n_neighbors, idhas)
            words.append(word)
            point["cluster_id"] = idhas
            point["in_cluster"] = True
            self.allwords = cp.append(self.allwords, point)

            centroids = Parallel(n_jobs=self.n_jobs)(
                delayed(self.model.get_word_vector)(w) for w in words)
            centroid_vector = cp.mean(centroids, axis=0)

            cluster = {'id': idhas,
                       'centroid': centroid_vector, 'words': words}
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

    def get_resumes_doc(self, top_n, n_clusters=4, tokens=None, tf_idf_dict=None):
        clusters, tf_idf_clusters = self.get_clusters(
            tokens=tokens, tf_idf_dict=tf_idf_dict)

        # Calcul des scores des clusters et sélection des meilleurs
        cluster_scores = tf_idf_clusters.sum(axis=0)
        best_cluster_indices = cp.argsort(cluster_scores)[
            ::-1][:n_clusters]  # Meilleurs clusters

        # Scores des documents basés sur les meilleurs clusters
        document_scores = tf_idf_clusters[:, best_cluster_indices].sum(axis=1)
        best_document_indices = cp.argsort(document_scores)[
            ::-1][:top_n]  # Meilleurs documents

        return best_document_indices

    def get_clusters(self, tokens, tf_idf_dict):
        self.clusters = []
        all_words = [{"word": token, "in_cluster": False,
                      "cluster_id": None} for token in tokens]

        self.allwords = all_words
        for word in tokens:
            self.add_word(word)
        tf_idf_clusters = self.build_tf_idf_vector_of_cluster(
            self.clusters, tf_idf_dict)
        return self.clusters, tf_idf_clusters

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




if __name__ == '__main__':

    process = TextPreprocessor()
    text = """
    
        Artificial intelligence, often referred to as AI, encompasses a broad range of applications, from natural language processing to computer vision. Machine learning, a subset of AI, focuses on the development of algorithms that enable computers to learn from and make predictions based on data. Together, these fields contribute to the rise of smart systems that can automate tasks and provide insights that were previously unattainable.

        Moreover, the integration of AI into everyday life is becoming increasingly seamless. Smart home devices, such as voice-activated assistants, exemplify how technology can simplify daily routines. In healthcare, AI algorithms assist in diagnosing diseases more accurately and quickly, leading to better patient outcomes.

        As we continue to explore the possibilities of these technologies, ethical considerations also come to the forefront. The responsible use of AI is crucial to ensure that innovation benefits society as a whole. Discussions around data privacy, algorithmic bias, and the impact of automation on employment are essential as we navigate this rapidly evolving landscape.

        In summary, the intersection of artificial intelligence, machine learning, and data analytics is revolutionizing industries and enhancing our lives. As we embrace these innovations, it is imperative to balance progress with ethical responsibility.
        
        """

    # Appel de la fonction
    import fasttext
    ft = fasttext.load_model('cc.en.300.bin')
    from nltk.tokenize import sent_tokenize
    documents = sent_tokenize(text)
    t, tf = process.preprocess(documents)
    # print(t)
    dbscan = DBSCAN(eps=0.4, model=ft, n_jobs=8)
    resume = dbscan.get_resumes_doc(
        top_n=2, n_clusters=3, tokens=t, tf_idf_dict=tf)

    """  c, c_t = dbscan.get_clusters(t, tf)
    print(c, c_t) """
    # print(documents[resume[0]])
