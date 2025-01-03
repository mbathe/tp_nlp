<<<<<<< HEAD
import textacy
import glob
import os
from tqdm import tqdm
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from rake_nltk import Rake
from nltk.tokenize import word_tokenize
import spacy
from dotenv import load_dotenv
load_dotenv()


class Processing:
    def __init__(self, log_file=None, corpus_dir="preprocessed", corpus_file="corpus.txt", mean_score=False, n_grams=2, min_n_gram_freq=2):
        self.log_file = log_file
        self.corpus_dir = corpus_dir
        self.tokens = set()
        self.corpus = ""
        self.corpus_file = corpus_file
        self.tf_idf_dict = []
        self.document_token = {}
        self.number_of_tokens = 0
        self.documents = []
        self.vec = None
        self.n_grams = n_grams
        self.mean_score = mean_score
        self.min_n_gram_freq = min_n_gram_freq

    def process_corpus(self):
        corpus_text = ""
        for file in tqdm(glob.glob(os.path.join(os.getenv('TXT_FOLDER2'), "*.txt"))):
            with open(file, "r", encoding="utf-8") as f:
                corpus_text += " "+f.read().strip().lower()

        self.documents = sent_tokenize(corpus_text)
        # self.corpus = corpus_text
        stop_words = set(stopwords.words('english'))
        document_count = len(self.documents)
        token_doc_count = defaultdict(int)
        nlp = spacy.load("en_core_web_sm")
        self.tf_idf_dict = [{} for i in range(len(self.documents))]
        for i, full_content in enumerate(tqdm(self.documents)):
            content = re.sub(
                r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)',
                "",
                full_content
            )

            doc = nlp(content)
            ngrams = list(textacy.extract.basics.ngrams(
                doc, self.n_grams, min_freq=self.min_n_gram_freq))
            target_pos = {"VERB", "ADJ", "NUM", "ADP", "PROPN",
                          "PRON", "PUNCT", "SCONJ", "SYM", "ADV", "SPACE", "AUX", "CONJ", "SYM", "PUNCT", "SCONJ"}
            target_tags = {"VB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                           "CD", "PRP", "PRP$", "DT", "IN", "CC", "UH", "SYM", "."}
            tokens = {token.lemma_ for token in doc if len(
                token.text) > 3 and token.text not in stop_words and token.pos_ not in target_pos and token.tag_ not in target_tags}
            tokens.update(set([str(ngram) for ngram in ngrams]))
            self.tokens.update(tokens)
            number_content_words = len(full_content.split())+len(set(ngrams))
            for token in tokens:
                token_doc_count[token] += 1
                self.tf_idf_dict[i][token] = (full_content.count(
                    token) / number_content_words if number_content_words > 1 else 0)

        for i in tqdm(range(document_count)):
            for token in self.tf_idf_dict[i]:
                self.tf_idf_dict[i][token] *= np.log(
                    document_count / (token_doc_count[token] + 1))
                # self.vec[i, tokens_index[token]] += self.tf_idf_dict[i][token]
    # .....

    def get_key_words(self):
        dictionnaire = dict(
            zip(self.tokens, np.zeros(len(self.tokens))))
        token_count_doc = dict(zip(self.tokens, np.ones(len(self.tokens))))
        for i in tqdm(range(len(self.tf_idf_dict))):
            for k in self.tf_idf_dict[i].keys():
                dictionnaire[k] += self.tf_idf_dict[i][k]
                if self.mean_score:
                    token_count_doc[k] += 1
        if self.mean_score:
            for k in tqdm(list(token_count_doc.keys())):
                dictionnaire[k] = dictionnaire[k]/token_count_doc[k]
        dictionnaire = dict(sorted(dictionnaire.items(),
                                   key=lambda item: item[1], reverse=True))
        return dictionnaire

    def compare_tokens_found(self):
        # Lire le corpus
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            corpus_text = f.read().strip().lower()

        r = Rake()

        # To get keyword phrases ranked highest to lowest.
        corpus_text = re.sub(
            r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\' ]+)',
            "",
            corpus_text
        )
        r.extract_keywords_from_text(corpus_text)
        return r.get_ranked_phrases_with_scores()

    def tfidf_filter(self):
        vectorizer = TfidfVectorizer()
        for i, d in enumerate(tqdm(self.documents)):
            tokens = word_tokenize(d)
            tokens = (' ').join([t.lower() for t in tokens
                                 if len(t) >= 3
                                 and (t.isalpha() or t in r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")
                                 and t.lower() not in stopwords.words('english')
                                 and "http" not in t.lower()
                                 ])
        X = vectorizer.fit_transform(tokens)
        tfidf_values = np.array(X.mean(axis=0))[0]
        median_tfidf = np.quantile(tfidf_values, 0.5)
        mask = tfidf_values > median_tfidf
        words_to_keep = vectorizer.get_feature_names_out()[mask]
        # print(type(words_to_keep))
        return words_to_keep
=======
import numpy as np


from collections import defaultdict
import cupy as cp
from collections import defaultdict
from nltk.corpus import stopwords
import re
import textacy
import spacy
import torch
from torch import amp
from concurrent.futures import ProcessPoolExecutor
import math


class TextPreprocessor:
    def __init__(self, batch_size=32):
        """
        Initialize with GPU support
        batch_size: Number of documents to process in parallel on GPU
        """
        spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000
        self.stop_words = set(stopwords.words('english'))
        self.nlp.max_length = 2_000_000
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
    from nltk.tokenize import sent_tokenize
    # Appel de la fonction
    tf_idf_keyword_process = Processing()
    documents = sent_tokenize(text)
    t, tf = process.preprocess(documents)
    top_doc_indices_keywords = tf_idf_keyword_process.get_resume_docs_index(
        top_n=2, number_key_words=30, tokens=t, tf_idf_dict=tf)

    # print(documents[top_doc_indices_keywords[0]])
>>>>>>> aabe059be106b5e4ac00675c900785ded16a91c5
