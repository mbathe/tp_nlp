import numpy as np


from collections import defaultdict


import numpy as np
import cupy as cp
from collections import defaultdict
from nltk.corpus import stopwords
import re
import textacy
import spacy
import torch
from torch.cuda import amp
from concurrent.futures import ThreadPoolExecutor
import math
from nltk.tokenize import sent_tokenize


class TextPreprocessor:
    def __init__(self, batch_size=32):
        """
        Initialize with GPU support
        batch_size: Number of documents to process in parallel on GPU
        """
        # Charge spaCy avec support CUDA
        spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Préchargement des filtres sur GPU
        self.target_pos = {"VERB", "ADJ", "NUM", "ADP", "PROPN", "PRON",
                           "PUNCT", "SCONJ", "SYM", "ADV", "SPACE", "AUX", "CONJ"}
        self.target_tags = {"VB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                            "CD", "PRP", "PRP$", "DT", "IN", "CC", "UH", "SYM", "."}

    def clean_document_batch(self, batch):
        """Nettoie un batch de documents en parallèle sur GPU"""
        pattern = r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)'

        # Convertir le batch en tenseur CUDA
        docs_tensor = torch.tensor([ord(c) for doc in batch for c in doc],
                                   device=self.device)

        # Utiliser torch.regex sur GPU (simulation - torch n'a pas de regex natif)
        cleaned_batch = [re.sub(pattern, "", doc) for doc in batch]
        return cleaned_batch

    def process_batch_gpu(self, batch, ngrams):
        """Traite un batch de documents sur GPU"""
        # Nettoyage du batch
        cleaned_batch = self.clean_document_batch(batch)

        # Traitement spaCy avec CUDA
        docs = list(self.nlp.pipe(cleaned_batch, batch_size=len(batch)))

        results = []
        for i, doc in enumerate(docs):
            # Extraction des tokens sur GPU
            with amp.autocast():
                tokens = {token.lemma_ for token in doc
                          if len(token.text) > 3
                          and token.text not in self.stop_words
                          and token.pos_ not in self.target_pos
                          and token.tag_ not in self.target_tags}

                # Extraction des n-grams pour ce document
                n_grams_doc = {ng for ng in ngrams if ng in cleaned_batch[i]}
                tokens.update(n_grams_doc)

                # Calcul des statistiques du document
                number_content_words = len(
                    cleaned_batch[i].split()) + len(n_grams_doc)

                # Calcul des fréquences de termes sur GPU
                tf_dict = {}
                if number_content_words > 1:
                    # Conversion en tenseurs CUDA pour le calcul vectorisé
                    doc_tensor = torch.tensor([ord(c) for c in cleaned_batch[i]],
                                              device=self.device)
                    for token in tokens:
                        token_tensor = torch.tensor([ord(c) for c in token],
                                                    device=self.device)
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

        # Traitement par batches sur GPU
        all_tokens = set()
        tf_idf_dict = [{} for _ in range(document_count)]
        token_doc_count = defaultdict(int)

        # Division en batches pour le GPU
        num_batches = math.ceil(len(documents) / self.batch_size)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(documents))
            batch = documents[start_idx:end_idx]

            # Traitement du batch sur GPU
            batch_results = self.process_batch_gpu(batch, ngrams)

            # Agrégation des résultats
            for doc_idx, tokens, tf_dict in batch_results:
                actual_idx = start_idx + doc_idx
                all_tokens.update(tokens)
                tf_idf_dict[actual_idx] = tf_dict
                for token in tokens:
                    token_doc_count[token] += 1

        # Calcul final TF-IDF sur GPU
        doc_count_tensor = cp.array(document_count)
        for i in range(document_count):
            doc_dict = tf_idf_dict[i]
            if doc_dict:
                # Conversion en tenseurs CuPy pour calcul vectorisé
                token_counts = cp.array([token_doc_count[token]
                                        for token in doc_dict])
                tf_values = cp.array(list(doc_dict.values()))

                # Calcul IDF vectorisé sur GPU
                idf = cp.log(doc_count_tensor / cp.maximum(token_counts, 1))
                tf_idf = tf_values * idf

                # Mise à jour du dictionnaire
                tf_idf_dict[i] = {token: float(score)
                                  for token, score in zip(doc_dict.keys(), tf_idf)}

        return all_tokens, tf_idf_dict

    def __del__(self):
        """Libération de la mémoire GPU"""
        # torch.cuda.empty_cache()


class Processing:
    def __init__(self, log_file=None, corpus_dir="preprocessed", mean_score=False, n_grams=2, min_n_gram_freq=2):
        self.log_file = log_file
        self.corpus_dir = corpus_dir
        self.tokens = set()
        self.corpus = ""
        self.tf_idf_dict = []
        self.document_token = {}
        self.number_of_tokens = 0
        self.vec = None
        self.n_grams = n_grams
        self.mean_score = mean_score
        self.min_n_gram_freq = min_n_gram_freq

    def get_key_words(self, tokens, tf_idf_dict):
        """
        Calculates the TF-IDF scores for each token in the corpus and returns a dictionary of tokens and their scores.
        If the 'mean_score' parameter is set to True, the function calculates the mean TF-IDF score for each token across all documents.
        """
        # Using defaultdict for automatic initialization
        score_dict = defaultdict(float)
        token_count = defaultdict(int)

        # Aggregate scores and counts
        for doc_scores in tf_idf_dict:
            for token, score in doc_scores.items():
                score_dict[token] += score
                token_count[token] += 1

        # Calculate mean scores if required
        if self.mean_score:
            for token in score_dict.keys():
                score_dict[token] /= token_count[token]

        # Sort dictionary by scores in descending order
        sorted_scores = dict(
            sorted(score_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores

    def get_resume_docs_index(self, top_n, tokens, tf_idf_dict, number_key_words=10):
        """
        Generates a summary of the top 'top_n' documents based on their relevance to a set of key words.
        """
        docs_scores = np.zeros(len(tf_idf_dict))
        tokens_tf_idf_scores = self.get_key_words(tokens, tf_idf_dict)
        key_words = list(tokens_tf_idf_scores.keys())[:number_key_words]
        # Convert to set for faster membership testing
        key_words_set = set(key_words)

        # Calculate scores for each document
        for i, doc_scores in enumerate(tf_idf_dict):
            doc_score = sum(
                tokens_tf_idf_scores[token] for token in doc_scores if token in key_words_set)
            docs_scores[i] = doc_score

        # Get indices of top documents
        docs_indices = np.argsort(docs_scores)[::-1][:top_n]
        return docs_indices


if __name__ == '__main__':
    process = TextPreprocessor()
    text = """
       Artificial intelligence, often referred to as AI, encompasses a broad range of applications, from natural language processing to computer vision. Machine learning, a subset of AI, focuses on the development of algorithms that enable computers to learn from and make predictions based on data. Together, these fields contribute to the rise of smart systems that can automate tasks and provide insights that were previously unattainable.

    Moreover, the integration of AI into everyday life is becoming increasingly seamless. Smart home devices, such as voice-activated assistants, exemplify how technology can simplify daily routines. In healthcare, AI algorithms assist in diagnosing diseases more accurately and quickly, leading to better patient outcomes.

    As we continue to explore the possibilities of these technologies, ethical considerations also come to the forefront. The responsible use of AI is crucial to ensure that innovation benefits society as a whole. Discussions around data privacy, algorithmic bias, and the impact of automation on employment are essential as we navigate this rapidly evolving landscape.

    In summary, the intersection of artificial intelligence, machine learning, and data analytics is revolutionizing industries and enhancing our lives. As we embrace these innovations, it is imperative to balance progress with ethical responsibility.
        """

    # Appel de la fonction
    tf_idf_keyword_process = Processing()
    documents = sent_tokenize(text)
    t, tf = process.preprocess(documents)
    top_doc_indices_keywords = tf_idf_keyword_process.get_resume_docs_index(
        top_n=2, number_key_words=30, tokens=t, tf_idf_dict=tf)

    # print(documents[top_doc_indices_keywords[0]])
