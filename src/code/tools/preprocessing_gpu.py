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
    def __init__(self, batch_size=128):
        """
        
        Initialize with GPU support
        batch_size: Number of documents to process in parallel on GPU
        
        
        """
        spacy.require_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000
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
