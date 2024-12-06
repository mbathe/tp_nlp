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


class TF_IDF_DOCUMENTS:
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
        for file in tqdm(glob.glob(os.path.join(os.getenv('TXT_FOLDER2'), "*.txt")), desc="constructing corpus"):
            with open(file, "r", encoding="utf-8") as f:
                corpus_text += " "+f.read().strip().lower()

        self.documents = sent_tokenize(corpus_text)
        # self.corpus = corpus_text
        stop_words = set(stopwords.words('english'))
        document_count = len(self.documents)
        token_doc_count = defaultdict(int)
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 2_000_000
        self.tf_idf_dict = [{} for i in range(len(self.documents))]
        for i, full_content in enumerate(tqdm(self.documents, desc="Processing documents")):
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
        for i in tqdm(range(len(self.tf_idf_dict)), desc="Calcul des scores"):
            for k in self.tf_idf_dict[i].keys():
                dictionnaire[k] += self.tf_idf_dict[i][k]
                if self.mean_score:
                    token_count_doc[k] += 1
        if self.mean_score:
            for k in tqdm(list(token_count_doc.keys()), desc="Calcul des moyennes des scores"):
                dictionnaire[k] = dictionnaire[k]/token_count_doc[k]
        dictionnaire = dict(sorted(dictionnaire.items(),
                                   key=lambda item: item[1], reverse=True))
        return dictionnaire

    def get_resume_docs(self, top_n):
        docs_scores = np.zeros(len(self.documents))
        tokens_tf_idf_scores = self.get_key_words()
        for i in range(len(self.documents)):
            doc_score = 0
            for token in self.tf_idf_dict[i].keys():
                if token in self.tokens:
                    doc_score += tokens_tf_idf_scores[token]
            docs_scores[i] = doc_score
        docs_indices = np.argsort(docs_scores)[::-1][:top_n]
        return docs_indices, np.array([self.documents[i] for i in docs_indices])

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
        return words_to_keep
