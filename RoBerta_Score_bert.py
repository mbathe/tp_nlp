#!pip install transformers
#!pip install torch
#!pip install spacy
#!python -m spacy download en_core_web_sm
#!pip install fsspec==2024.10.0 --no-deps
#!pip install bert_score
#!pip install evaluate

import re
import spacy
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import pandas as pd
from evaluate import load

class DocumentProcessor:
    def __init__(self, n_grams=2, min_n_gram_freq=1, stop_words=None):
        self.n_grams = n_grams
        self.min_n_gram_freq = min_n_gram_freq
        if stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = set(stop_words)

        # Charger le modèle spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("Modèle spaCy chargé avec succès.")
        except OSError:
            print("Erreur : Le modèle spaCy 'en_core_web_sm' n'est pas installé. Installation en cours...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
            print("Modèle spaCy installé et chargé avec succès.")

        # Charger le modèle et le tokenizer RoBERTa
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = RobertaModel.from_pretrained('roberta-base')
        print("Modèle RoBERTa chargé avec succès.")

        # Charger BERTScore
        self.bertscore = load("bertscore")
        print("BERTScore chargé avec succès.")

    def nettoyer_texte(self, text):
        """
        Nettoie le texte en supprimant les hashtags, mentions, emails, URLs, et caractères non pertinents.

        :param text: Texte brut à nettoyer
        :return: Texte nettoyé
        """
        # Regex pour supprimer les hashtags, mentions, emails, URLs et caractères non pertinents
        text = re.sub(r'(#\S+|@\S+|\S*@\S*\s?|http\S+|[^A-Za-z0-9\'\- \.\,\!\?\;]+)', "", text)

        # Suppression des espaces multiples et des lignes vides
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def segmenter_phrases(self, text):
        """
        Segmente le texte nettoyé en phrases en utilisant spaCy.

        :param text: Texte nettoyé
        :return: Liste de phrases
        """
        doc = self.nlp(text)
        phrases = [sent.text.strip() for sent in doc.sents]
        return phrases

    def encode_sentence(self, sentence):
        """
        Encode une phrase en utilisant RoBERTa.

        :param sentence: Phrase à encoder
        :return: Encodage de la phrase sous forme de numpy array (hidden_size,)
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()

    def calculer_similarites(self, phrases, text_encoding):
        """
        Calcule la similarité cosinus entre chaque phrase et le texte entier.

        :param phrases: Liste de phrases
        :param text_encoding: Encodage du texte entier (hidden_size,)
        :return: Liste de tuples (phrase, similarité)
        """
        similarities = []
        for phrase in phrases:
            if not phrase:  # Ignorer les phrases vides
                continue
            phrase_encoding = self.encode_sentence(phrase)
            similarity = cosine_similarity([text_encoding], [phrase_encoding]).item()
            similarities.append((phrase, similarity))
        return similarities

    def calculer_similarites_entre_phrases(self, phrases_encodées):
        """
        Calcule la similarité cosinus entre chaque paire de phrases.

        :param phrases_encodées: Liste d'encodages de phrases (list of numpy arrays, each (hidden_size,))
        :return: Matrice de similarité (n_phrases, n_phrases)
        """
        if not phrases_encodées:
            return np.array([])

        # Empiler les encodages en une matrice 2D (n_phrases, hidden_size)
        phrases_encodées_stacked = np.vstack(phrases_encodées)
        similarity_matrix = cosine_similarity(phrases_encodées_stacked)
        return similarity_matrix

    def calculer_scores(self, phrases, text):
        """
        Calcule les scores de similarité BERTScore entre chaque phrase et le texte complet.

        :param phrases: Liste de phrases
        :param text: Texte complet
        :return: DataFrame trié des phrases et de leurs scores
        """
        scores = []
        for phrase in phrases:
            if not phrase:  # Ignorer les phrases vides
                continue
            results = self.bertscore.compute(predictions=[phrase], references=[text], lang="en")
            score = results['f1'][0]  # Utiliser 'precision', 'recall' ou 'f1'
            scores.append((phrase, score))

        df = pd.DataFrame(scores, columns=['Phrase', 'Score'])
        df_sorted = df.sort_values(by='Score', ascending=False)
        return df_sorted

    def process_document(self, content):
        """
        Traite un document individuel en nettoyant le texte, en segmentant en phrases,
        et en calculant les similarités selon trois méthodes, ainsi que les scores BERTScore.

        :param content: Contenu textuel du document à traiter
        :return: Dictionnaire contenant les résultats des trois méthodes et le DataFrame des scores BERTScore
        """
        # 1. Nettoyage du texte
        text_propre = self.nettoyer_texte(content)
        print("\nExtrait du texte nettoyé (500 premiers caractères) :")
        print(text_propre[:500])

        # 2. Segmentation en phrases
        phrases = self.segmenter_phrases(text_propre)
        print(f"\nNombre de phrases extraites : {len(phrases)}")
        if phrases:
            print("\nQuelques phrases extraites :")
            for i, phrase in enumerate(phrases[:10]):
                print(f"Phrase {i+1} :", phrase)
        else:
            print("\nAucune phrase n'a été extraite.")

        # 3. Encodage du texte entier
        if text_propre:
            text_encoding = self.encode_sentence(text_propre)
        else:
            print("Le texte après nettoyage est vide. Impossible de continuer.")
            return {
                "method1": "",
                "method2": "",
                "hybrid": "",
                "bertscore_df": pd.DataFrame()
            }

        # Méthode 1 : Comparer chaque phrase avec le texte entier
        similarites_avec_texte = self.calculer_similarites(phrases, text_encoding)
        if similarites_avec_texte:
            phrase_most_relevant_method1 = max(similarites_avec_texte, key=lambda x: x[1])[0]
            print("\nMéthode 1 - La phrase la plus pertinente est :", phrase_most_relevant_method1)
        else:
            phrase_most_relevant_method1 = ""
            print("\nMéthode 1 - Aucune phrase pertinente trouvée.")

        # Méthode 2 : Comparer les phrases entre elles
        if phrases:
            phrases_encodées = [self.encode_sentence(phrase) for phrase in phrases if phrase]
            similarity_matrix = self.calculer_similarites_entre_phrases(phrases_encodées)
            if similarity_matrix.size > 0:
                centralite = similarity_matrix.sum(axis=1)
                indice_centrale = centralite.argmax()
                phrase_most_relevant_method2 = phrases[indice_centrale]
                print("Méthode 2 - La phrase la plus centrale est :", phrase_most_relevant_method2)
            else:
                phrase_most_relevant_method2 = ""
                print("Méthode 2 - Aucune similarité trouvée entre les phrases.")
        else:
            phrase_most_relevant_method2 = ""
            print("Méthode 2 - Aucune phrase trouvée pour comparaison.")

        # Méthode 3 : Approche Hybride
        # Étape 1 : Comparer chaque phrase avec le texte entier et sélectionner les top N
        N = 5
        top_phrases_method1 = sorted(similarites_avec_texte, key=lambda x: x[1], reverse=True)[:N]
        phrases_top_method1 = [phrase for phrase, score in top_phrases_method1]
        if phrases_top_method1:
            print(f"\nMéthode 3 - Top {N} phrases les plus pertinentes selon Méthode 1 :")
            for i, phrase in enumerate(phrases_top_method1, 1):
                print(f"Top Phrase {i} :", phrase)
        else:
            print("\nMéthode 3 - Aucune phrase pertinente trouvée pour l'approche hybride.")

        # Étape 2 : Comparer ces top phrases entre elles pour trouver la plus centrale
        if phrases_top_method1:
            phrases_top_encodées = [self.encode_sentence(phrase) for phrase in phrases_top_method1]
            similarity_matrix_top = self.calculer_similarites_entre_phrases(phrases_top_encodées)
            if similarity_matrix_top.size > 0:
                centralite_top = similarity_matrix_top.sum(axis=1)
                indice_centrale_top = centralite_top.argmax()
                phrase_most_relevant_hybride = phrases_top_method1[indice_centrale_top]
                print("Méthode 3 - La phrase la plus pertinente (hybride) est :", phrase_most_relevant_hybride)
            else:
                phrase_most_relevant_hybride = ""
                print("Méthode 3 - Aucune similarité trouvée parmi les top phrases.")
        else:
            phrase_most_relevant_hybride = ""
            print("Méthode 3 - Aucune phrase pertinente trouvée pour l'approche hybride.")

        # Calcul des scores BERTScore
        print("\nCalcul des scores BERTScore pour chaque phrase...")
        bertscore_df = self.calculer_scores(phrases, text_propre)
        print("\nDataFrame des scores BERTScore :")
        print(bertscore_df.head(10))

        # Évaluation des méthodes avec BERTScore
        print("\nÉvaluation des méthodes avec BERTScore...")

        methods = {
            'method1': phrase_most_relevant_method1,
            'method2': phrase_most_relevant_method2,
            'hybrid': phrase_most_relevant_hybride
        }

        bertscore_results = {}

        for method_name, method_output in methods.items():
            if method_output:
                results = self.bertscore.compute(predictions=[method_output], references=[text_propre], lang="en")
                score = results['f1'][0]
                bertscore_results[method_name] = score
                print(f"{method_name} BERTScore F1: {score}")
            else:
                bertscore_results[method_name] = None
                print(f"{method_name} n'a pas produit de phrase pour évaluation.")

        # Déterminer la méthode avec le score le plus élevé
        best_method = max(bertscore_results, key=lambda k: bertscore_results[k] if bertscore_results[k] is not None else -1)

        print(f"\nLa meilleure méthode selon BERTScore est : {best_method} avec un score de {bertscore_results[best_method]}")

        # Retourner les résultats sous forme de dictionnaire
        return {
            "method1": phrase_most_relevant_method1,
            "method2": phrase_most_relevant_method2,
            "hybrid": phrase_most_relevant_hybride,
            "bertscore_df": bertscore_df,
            "bertscore_methods": bertscore_results,
            "best_method": best_method
        }

    def lire_fichier(self, chemin_fichier):
        """
        Lit le contenu d'un fichier texte.

        :param chemin_fichier: Chemin vers le fichier texte
        :return: Contenu du fichier sous forme de chaîne de caractères
        """
        try:
            with open(chemin_fichier, 'r', encoding='utf-8') as fichier:
                texte = fichier.read()
            print("Fichier chargé avec succès.")
            return texte
        except FileNotFoundError:
            print(f"Erreur : Le fichier à l'emplacement {chemin_fichier} n'a pas été trouvé.")
            return ""

# Exemple d'utilisation de la classe DocumentProcessor
if __name__ == "__main__":
    # Initialiser le processeur de document
    processor = DocumentProcessor(n_grams=2, min_n_gram_freq=1, stop_words=['example', 'stopword'])

    # Charger le contenu du fichier
    file_path = '/content/5.txt'  # Remplace par ton chemin de fichier
    content = processor.lire_fichier(file_path)

    # Traiter le document
    resultats = processor.process_document(content)

    # Afficher les résultats
    print("\nRésultats des Méthodes de Similarité :")
    print(f"Méthode 1 - Phrase la plus pertinente : {resultats['method1']}")
    print(f"Méthode 2 - Phrase la plus centrale : {resultats['method2']}")
    print(f"Méthode 3 - Phrase la plus pertinente (hybride) : {resultats['hybrid']}")

    print("\nÉvaluation des méthodes avec BERTScore :")
    for method, score in resultats['bertscore_methods'].items():
        print(f"{method} : {score}")

    print(f"\nLa meilleure méthode selon BERTScore est : {resultats['best_method']}")

    # Sauvegarde du DataFrame des scores BERTScore sous forme de fichier CSV
    resultats['bertscore_df'].to_csv('dataframe.csv', index=False)
    print("\nLe fichier CSV a été sauvegardé sous le nom 'dataframe.csv'")
