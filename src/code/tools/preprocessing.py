import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
import re
import textacy
from nltk.tokenize import sent_tokenize
import spacy
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools


class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000
        self.stop_words = set(stopwords.words('english'))
        self.target_pos = {"VERB", "ADJ", "NUM", "ADP", "PROPN", "PRON",
                           "PUNCT", "SCONJ", "SYM", "ADV", "SPACE", "AUX", "CONJ"}
        self.target_tags = {"VB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS",
                            "CD", "PRP", "PRP$", "DT", "IN", "CC", "UH", "SYM", "."}

    def clean_document(self, content):
        """Clean individual document content."""
        return re.sub(
            r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9]\'\'|\d+|<[^>]*>|[^A-Za-z0-9\'\- ]+)',
            "",
            content
        )

    def process_document(self, args):
        """Process a single document with its index and n-grams."""
        document, i, ngrams = args

        # Clean the document
        cleaned_content = self.clean_document(document)

        # Process with spaCy
        doc = self.nlp(cleaned_content)

        # Extract tokens
        tokens = {token.lemma_ for token in doc
                  if len(token.text) > 3
                  and token.text not in self.stop_words
                  and token.pos_ not in self.target_pos
                  and token.tag_ not in self.target_tags}

        # Extract n-grams for this document
        n_grams_doc = {ng for ng in ngrams if ng in cleaned_content}
        tokens.update(n_grams_doc)

        # Calculate document statistics
        number_content_words = len(document.split()) + len(n_grams_doc)

        # Calculate term frequencies
        tf_dict = {}
        for token in tokens:
            tf_dict[token] = document.count(
                token) / number_content_words if number_content_words > 1 else 0

        return i, tokens, tf_dict

    def preprocess(self, documents, n_grams=2, min_n_gram_freq=2):
        """Processes a corpus of documents in parallel."""
        document_count = len(documents)

        # Extract n-grams from the entire corpus
        joined_docs = "\n".join(documents)
        doc = self.nlp(joined_docs)
        ngrams = {str(n_gram) for n_gram in textacy.extract.basics.ngrams(
            doc, n_grams, min_freq=min_n_gram_freq)}

        # Prepare arguments for parallel processing
        process_args = [(doc, i, ngrams) for i, doc in enumerate(documents)]

        # Process documents in parallel
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.process_document, process_args)

        # Initialize data structures
        all_tokens = set()
        tf_idf_dict = [{} for _ in range(document_count)]
        token_doc_count = defaultdict(int)

        # Combine results
        for i, tokens, tf_dict in results:
            all_tokens.update(tokens)
            tf_idf_dict[i] = tf_dict
            for token in tokens:
                token_doc_count[token] += 1

        for i in range(document_count):
            for token in tf_idf_dict[i]:
                tf_idf_dict[i][token] *= np.log(document_count / (
                    token_doc_count[token])) if token_doc_count[token] > 1 else 1

        return all_tokens, tf_idf_dict


if __name__ == '__main__':
    process = TextPreprocessor()
    text = """
       Artificial intelligence, often referred to as AI, encompasses a broad range of applications, from natural language processing to computer vision. Machine learning, a subset of AI, focuses on the development of algorithms that enable computers to learn from and make predictions based on data. Together, these fields contribute to the rise of smart systems that can automate tasks and provide insights that were previously unattainable.

    Moreover, the integration of AI into everyday life is becoming increasingly seamless. Smart home devices, such as voice-activated assistants, exemplify how technology can simplify daily routines. In healthcare, AI algorithms assist in diagnosing diseases more accurately and quickly, leading to better patient outcomes.

    As we continue to explore the possibilities of these technologies, ethical considerations also come to the forefront. The responsible use of AI is crucial to ensure that innovation benefits society as a whole. Discussions around data privacy, algorithmic bias, and the impact of automation on employment are essential as we navigate this rapidly evolving landscape.

    In summary, the intersection of artificial intelligence, machine learning, and data analytics is revolutionizing industries and enhancing our lives. As we embrace these innovations, it is imperative to balance progress with ethical responsibility.
        """

    # Appel de la fonction

    documents = sent_tokenize(text)
    t, tf = process.preprocess(documents)
    # print(t, tf)
