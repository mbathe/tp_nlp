from dotenv import dotenv_values
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import glob
import os
import re
import numpy as np
current_working_directory = os.getcwd()


def tfidf_filter_corpus():
    corpus_text = ""
    for file in tqdm(glob.glob(current_working_directory +
                               dotenv_values(".env")['TXT_FOLDER2'] + '*.txt')):
        with open(file, "r", encoding="utf-8") as f:
            content = [re.sub(
                r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)',
                "", sentence

            )
                for sentence in f.read().lower().split('.')]

            print(content)
            input()
            corpus_text += " "+content

    documents = sent_tokenize(corpus_text)

    print(documents[0])

    words_to_keep = tfidf_filter(documents)
    print(words_to_keep)


def tfidf_filter(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    tfidf_values = np.array(X.mean(axis=0))[0]
    median_tfidf = np.quantile(tfidf_values, 0.5)
    mask = tfidf_values > median_tfidf
    words_to_keep = vectorizer.get_feature_names_out()[mask]
    print(type(words_to_keep))
    return words_to_keep


tfidf_filter_corpus()
