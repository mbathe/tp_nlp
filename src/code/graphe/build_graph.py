import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
import re


class Graphe:
    def __init__(self, documents, custom_stop_words=None):
        self.documents = documents
        self.terms = []
        self.co_occurrence = None
        self.distance_matrix = None
        self.graph = None
        # Liste des mots personnalisés à exclure
        self.custom_stop_words = custom_stop_words if custom_stop_words else list(stopwords.words(
            'english'))

    def preprocess(self):
        self.documents = [re.sub(
            r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)', "", doc) for doc in self.documents]
        vectorizer = CountVectorizer(ngram_range=(
            1, 2), stop_words=self.custom_stop_words)
        X = vectorizer.fit_transform(self.documents)
        self.terms = vectorizer.get_feature_names_out()
        self.co_occurrence = (X.T @ X).toarray()
        np.fill_diagonal(self.co_occurrence, 0)
        # print(X.shape, self.terms.shape, self.co_occurrence.shape)

    def compute_distance_matrix(self):
        self.distance_matrix = np.zeros(self.co_occurrence.shape)
        for i in tqdm(range(self.co_occurrence.shape[0])):
            for j in range(self.co_occurrence.shape[1]):
                if self.co_occurrence[i, j] > 0:
                    self.distance_matrix[i, j] = 1 / self.co_occurrence[i, j]
                else:
                    self.distance_matrix[i, j] = np.inf

    def build_graph(self):
        self.graph = nx.Graph()
        for i in tqdm(range(len(self.terms))):
            for j in range(len(self.terms)):
                if i != j:
                    self.graph.add_edge(
                        self.terms[i], self.terms[j], weight=self.distance_matrix[i, j])

    def detect_keywords(self):
        keyword_scores = {}
        for i, term in enumerate(self.terms):
            neighbors = list(self.graph.neighbors(term))
            if neighbors:
                avg_distance = np.sum(
                    [self.distance_matrix[i, self.terms.tolist().index(n)] for n in neighbors])
                keyword_scores[term] = avg_distance

        # Trier par moyenne des distances
        keywords = sorted(keyword_scores.items(), key=lambda item: item[1])
        return keywords

    def summarize(self, num_sentences=2):
        sentences = []
        for doc in self.documents:
            sentences.extend(sent_tokenize(doc, language='english'))

        vectorizer = CountVectorizer(stop_words='english')
        X_sentences = vectorizer.fit_transform(sentences)
        co_occurrence_sent = (X_sentences.T @ X_sentences).toarray()
        np.fill_diagonal(co_occurrence_sent, 0)

        graph_sent = nx.Graph()
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j and co_occurrence_sent[i, j] > 0:
                    graph_sent.add_edge(
                        sentences[i], sentences[j], weight=1/co_occurrence_sent[i, j])

        centrality_sent = nx.degree_centrality(graph_sent)
        ranked_sentences = sorted(
            centrality_sent.items(), key=lambda item: item[1], reverse=True)

        summary = [sentence for sentence,
                   _ in ranked_sentences[:num_sentences]]
        return summary

    def analyze(self):
        self.preprocess()
        self.compute_distance_matrix()
        self.build_graph()
        return self.detect_keywords()
