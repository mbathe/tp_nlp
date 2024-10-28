import os
from src.code.TFIDF.pre_processing import Processing
from nltk.corpus import stopwords
from dotenv import load_dotenv
from src.code.graphe.build_graph import Graphe
from tqdm import tqdm
import glob
from nltk.tokenize import sent_tokenize
from src.code.bert.bert import Bert
# load_dotenv()



if __name__ == '__main__':
    document = ""
    for file in tqdm(glob.glob(os.path.join(os.getenv('TXT_FOLDER2'), "*.txt"))[0:50]):
        with open(file, "r", encoding="utf-8") as f:
            doc = f.read().strip().lower()
            if len(doc) > 10:
                document += doc + "\n"
    documents = sent_tokenize(document)

    bert = Bert(documents=documents)
    bert.processing()
    top_indices, summary = bert.get_resume(top_n=10)
    print(top_indices)

    preprocessing = Processing(
        log_file="", corpus_dir="processed", documents=documents)
    preprocessing.process_corpus()
    keys_words = preprocessing.get_key_words()
    if "also" in set(stopwords.words('english')):
        print("aloso" in set(stopwords.words('english')))
    # print(len(list(keys_words.keys())))
    print(list(keys_words.keys())[0:40])
    resume_indeces, resume_docs = preprocessing.get_resume_docs(top_n=10)
    print(resume_indeces)
