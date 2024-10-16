import os
from src.code.TFIDF.pre_processing import Processing
from nltk.corpus import stopwords
from dotenv import load_dotenv
load_dotenv()

if __name__ == '__main__':
    preprocessing = Processing(
        log_file="", corpus_dir="processed", corpus_file=os.getenv("CORPUS_FILE"))
    preprocessing.process_corpus()

    keys_words = preprocessing.get_key_words()
    if "also" in set(stopwords.words('english')):
        print("aloso" in set(stopwords.words('english')))
    # print(len(list(keys_words.keys())))
    print(list(keys_words.keys())[0:40])
