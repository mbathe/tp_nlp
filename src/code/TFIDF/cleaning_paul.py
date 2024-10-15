from tqdm import tqdm
import glob
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from collections import defaultdict


class Processing:
    def __init__(self, log_file=None, corpus_dir="preprocessed", corpus_file="corpus.txt"):
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

    def process_corpus(self):
        # Lire le corpus
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            corpus_text = f.read().strip().lower()

        self.documents = sent_tokenize(corpus_text)
        self.corpus = corpus_text
        tk = WordPunctTokenizer()
        stop_words = set(stopwords.words('english'))
        document_count = len(self.documents)
        token_doc_count = defaultdict(int)
        self.tf_idf_dict = [{} for i in range(len(self.documents))]
        for i, full_content in enumerate(tqdm(self.documents)):
            content = re.sub(
                r'(#\S+|@\S+|\S*@\S*\s?|http\S+|word01|word02|word03|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\' ]+)',
                "",
                full_content
            )
            tokens = {token for token in tk.tokenize(content) if len(
                token) > 3 and token not in stop_words}
            self.tokens.update(tokens)
            number_content_words = len(full_content.split())
            for token in tokens:
                token_doc_count[token] += 1
                self.tf_idf_dict[i][token] = (full_content.count(
                    token) / number_content_words if number_content_words > 1 else 0)

        print("Calculer les TF-IDF")
        tokens_index = dict(zip(self.tokens, range(len(self.tokens))))
       # self.vec = np.zeros((len(self.documents), len(self.tokens)))

        for i in tqdm(range(document_count)):
            for token in self.tf_idf_dict[i]:
                self.tf_idf_dict[i][token] *= np.log(
                    document_count / (token_doc_count[token] + 1))
                # self.vec[i, tokens_index[token]] += self.tf_idf_dict[i][token]

        print(self.tf_idf_dict[i])

    def tfidf_filter(self, corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        tfidf_values = np.array(X.mean(axis=0))[0]
        median_tfidf = np.quantile(tfidf_values, 0.5)
        mask = tfidf_values > median_tfidf
        words_to_keep = vectorizer.get_feature_names_out()[mask]
        # print(type(words_to_keep))
        return words_to_keep
