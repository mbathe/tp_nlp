import re
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from functools import lru_cache
from typing import List, Dict
import warnings

class DocumentProcessor:
    def __init__(self, n_grams=2, min_n_gram_freq=1, stop_words=None):
        self.n_grams = n_grams
        self.min_n_gram_freq = min_n_gram_freq
        self.stop_words = set(stop_words or [])
        self.device = 'cpu'
        self.batch_size = 32

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base', add_prefix_space=True)
            # Charger uniquement les couches nécessaires
            self.model = RobertaModel.from_pretrained(
                'roberta-base',
                output_hidden_states=True,
                output_attentions=False,
                add_pooling_layer=False  # Désactiver la couche de pooling non utilisée
            )
            self.model.eval()

    @staticmethod
    @lru_cache(maxsize=1024)
    def nettoyer_texte(text: str) -> str:
        """Version mise en cache du nettoyage de texte"""
        text = re.sub(
            r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9\'\- \.\,\!\?\;]+)', "", text)
        return re.sub(r'\s+', ' ', text).strip()

    def encode_sentences_batch(self, sentences: List[str]) -> np.ndarray:
        """Encode plusieurs phrases en une seule fois"""
        if not sentences:
            return np.array([])

        encodings = []
        with torch.no_grad():  # Désactiver le gradient pour l'inférence
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i:i + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                    add_special_tokens=True
                )

                outputs = self.model(**inputs)
                # Utiliser uniquement le dernier hidden state
                last_hidden_state = outputs.last_hidden_state
                # Moyenne sur la dimension des tokens
                batch_encodings = last_hidden_state.mean(dim=1)
                encodings.append(batch_encodings.numpy())

        return np.vstack(encodings) if encodings else np.array([])

    def process_document(self, documents: List[str]) -> Dict[str, str]:
        """
        Process a list of documents to find the most relevant sentences using different methods.

        This function applies three methods to determine the most relevant sentence:
        1. Highest similarity to the complete text
        2. Highest centrality (most similar to all other sentences)
        3. A hybrid approach combining the two methods

        Parameters:
        documents (List[str]): A list of strings, each representing a document or sentence to process.

        Returns:
        Dict[str, str]: A dictionary containing the most relevant sentences found by each method:
            - 'method1': The sentence with the highest similarity to the complete text
            - 'method2': The sentence with the highest centrality
            - 'hybrid': The most relevant sentence found using the hybrid approach

        In case of an error, it returns the first document (if available) for all methods.
        """
        # Nettoyage et préparation
        phrases_propres = [self.nettoyer_texte(
            doc) for doc in documents if doc]

        text_complet = " ".join(phrases_propres)

        # Encodage batch optimisé
        try:
            toutes_phrases_encodees = self.encode_sentences_batch(
                phrases_propres)
            text_complet_encode = self.encode_sentences_batch([text_complet])[
                0]

            # Calcul des similarités vectorisé
            similarites_texte = cosine_similarity(
                toutes_phrases_encodees,
                text_complet_encode.reshape(1, -1)
            ).flatten()

            # Méthode 1: Plus grande similarité
            indice_max_sim = np.argmax(similarites_texte)
            phrase_most_relevant_method1 = documents[indice_max_sim]

            # Méthode 2: Centralité
            similarity_matrix = cosine_similarity(toutes_phrases_encodees)
            centralite = similarity_matrix.sum(axis=1)
            indice_centrale = centralite.argmax()
            phrase_most_relevant_method2 = documents[indice_centrale]

            # Méthode hybride
            N = min(5, len(documents))
            top_indices = np.argpartition(similarites_texte, -N)[-N:]
            top_phrases = [documents[i] for i in top_indices]

            if top_phrases:
                top_phrases_encodees = self.encode_sentences_batch(top_phrases)
                similarity_matrix_top = cosine_similarity(top_phrases_encodees)
                centralite_top = similarity_matrix_top.sum(axis=1)
                indice_centrale_top = centralite_top.argmax()
                phrase_most_relevant_hybride = top_phrases[indice_centrale_top]
            else:
                phrase_most_relevant_hybride = ""

            return {
                'method1': phrase_most_relevant_method1,
                'method2': phrase_most_relevant_method2,
                'hybrid': phrase_most_relevant_hybride
            }

        except Exception as e:
            print(f"Erreur lors du traitement du document: {str(e)}")
            return {
                'method1': documents[0] if documents else "",
                'method2': documents[0] if documents else "",
                'hybrid': documents[0] if documents else ""
            }

    @staticmethod
    def lire_fichier(chemin_fichier: str) -> str:
        """Lecture de fichier avec gestion d'erreurs"""
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                return fichier.read()
        except FileNotFoundError:
            return ""


0
