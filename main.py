import os
from src.code.TFIDF.pre_processing import Processing
from nltk.corpus import stopwords
from dotenv import load_dotenv
from src.code.graphe.build_graph import Graphe
from tqdm import tqdm
import glob
# load_dotenv()

if __name__ == '__main__':
    """ preprocessing = Processing(
        log_file="", corpus_dir="processed", corpus_file=os.getenv("CORPUS_FILE"))
    preprocessing.process_corpus()

    keys_words = preprocessing.get_key_words()
    if "also" in set(stopwords.words('english')):
        print("aloso" in set(stopwords.words('english')))
    # print(len(list(keys_words.keys())))
    print(list(keys_words.keys())[0:40])
 """

    documents = []
    for file in tqdm(glob.glob(os.path.join(os.getenv('TXT_FOLDER2'), "*.txt"))[0:1]):
        with open(file, "r", encoding="utf-8") as f:
            doc = f.read().strip().lower()
            if len(doc) > 10:
                documents.append(doc)
    # print(documents[0])
    graphe = Graphe(documents)
    keywords = graphe.analyze()

    print("Mots-clés détectés :")
    for word, score in keywords:
        print(f"{word}: {score:.4f}")

    summary = graphe.summarize(num_sentences=1)
    print("\nRésumé des documents :")
    for sentence in summary:
        print(f"- {sentence}")
