from tqdm import tqdm
import re
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import numpy as np
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import os
import glob
import pandas as pd

current_working_directory = os.getcwd()


class Processing:
    def __init__(self):
        self.tokens = set()
        self.corpus = ""
        self.tf_idf_dict = []
        self.tf_idf_dataframe = pd.DataFrame([])
        self.document_token = {}
        self.number_of_tokens = 0
        self.sentences = []
        self.vec = None

    def process_corpus(self):
        # Créer le corpus avec des phrases
        for _, filename in enumerate(tqdm(glob.glob(current_working_directory+"\\src\\docfile\\txts\\*.txt"))):
            with open(filename, encoding="utf8") as f:
                document_text = f.read().strip().lower()
            self.corpus += document_text

        self.sentences = sent_tokenize(self.corpus)
        sentence_count = len(self.sentences)

        tk = WordPunctTokenizer()
        token_doc_count = defaultdict(int)
        self.tf_idf_dict = [{} for _ in range(len(self.sentences))]

        stop_words = set(stopwords.words('english'))

        for i, full_content in tqdm(enumerate(self.sentences)):

            content = re.sub(
                r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\' ]+)',
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

                # self.tf_idf_dataframe[i][token] = (full_content.count(
                #     token) / number_content_words if number_content_words > 1 else 0)

        for i in tqdm(range(sentence_count)):

            for token in self.tf_idf_dict[i]:

                self.tf_idf_dict[i][token] *= np.log(
                    sentence_count / (token_doc_count[token] + 1))

                # self.tf_idf_dataframe[i][token] *= np.log(
                #     sentence_count / (token_doc_count[token] + 1))

        # df = self.tf_idf_dataframe
        # display(df)

        return self
