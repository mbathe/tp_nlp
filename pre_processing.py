import tqdm
import glob
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Processing:
    def __init__(self, log_file=None, corpus_dir="preprocessed"):
        self.log_file = log_file
        self.corpus_dir = corpus_dir
        self.tokens = set()
        self.corpus = ""

    def clean_corpus(self):
        for i, fname in enumerate(tqdm(glob.glob(f"./{self.corpus_dir}/*.txt"))):
            f = open(fname, "r", encoding="utf-8")
            content = f.read().strip().lower()
            content = re.sub(r'#\S+', "", content)
            content = re.sub(r'@\S+', "", content)
            content = re.sub(r'\S*@\S*\s?', "", content)
            content = re.sub(r'http\S+', "", content)
            content = re.sub(r'word01|word02|word03', "", content)
            content = re.sub(r"[^A-Za-z0-9]''", "", content)
            content = re.sub(f'\d+', "", content)
            content = re.sub(r'<[^>]*>', "", content)
            content = re.sub("[^A-Za-z0-9|' ']+", "", content)
            self.tokens = self.tokens.union(set([token for token in WordPunctTokenizer(
                content) if len(token) > 3 and token not in stopwords]))
            self.corpus += " " + content

    def tfidf_filter(self, corpus):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        tfidf_values = np.array(X.mean(axis=0))[0]
        median_tfidf = np.quantile(tfidf_values, 0.5)
        mask = tfidf_values > median_tfidf
        words_to_keep = vectorizer.get_feature_names_out()[mask]
        print(type(words_to_keep))
        return words_to_keep
