from tqdm import tqdm
import torch.multiprocessing as mp
from src.code.tools.preprocessing_gpu import TextPreprocessor
from src.code.tools.metrics import get_bert_score
import fasttext.util
import numpy as np
import os
from src.code.TF_IDF_Fasttext.dbscan import DBSCAN
from src.code.TFIDF.pre_processing import Processing
from src.code.bert.bert import Bert
from dotenv import load_dotenv
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import cupy as cp
import gc
from nltk.tokenize import sent_tokenize
from src.code.Roberta.roberta import DocumentProcessor
load_dotenv()
ft = fasttext.load_model('cc.en.300.bin')

methodes = {
    "TF_IDF_KEY_WORD": 1,
    "BERT": 2,
    "TF_IDF_FASTTEXT": 3,
    "ROBERTA": 4
}


processor_roberta = DocumentProcessor(
    n_grams=2, min_n_gram_freq=1, stop_words=['example', 'stopword'])
tf_idf_keyword_process = Processing()
processor = TextPreprocessor()
tf_idf_fasttext_process = DBSCAN(eps=0.4, model=ft)


def process_tf_idf_key_word(documents, tokens, tf_idf_dict, top_n):
    top_doc_indices_keywords = tf_idf_keyword_process.get_resume_docs_index(
        top_n=top_n, number_key_words=30, tokens=tokens, tf_idf_dict=tf_idf_dict)
    return "TF_IDF_KEY_WORD", "\n".join(documents[top_doc_indices_keywords[0:top_n]])


def process_tf_idf_fasttext(documents, tokens, tf_idf_dict, top_n):
    best_resumes_index = tf_idf_fasttext_process.get_resumes_doc(
        top_n=2, n_clusters=3, tokens=tokens, tf_idf_dict=tf_idf_dict)
    return "TF_IDF_FASTTEXT", "\n".join(documents[best_resumes_index[0:top_n]])


def process_bert(documents, top_n):
    bert = Bert(documents=documents)
    bert.processing()
    top_doc_indices_bert, resume_doc_bert = bert.get_resume(top_n=top_n)
    return "BERT", "\n".join(documents[top_doc_indices_bert[0:top_n]])


def process_roberta(documents):
    resultats = processor_roberta.process_document(documents=documents)
    return "ROBERTA", resultats

def get_resume_by_method(documents, top_n, tokens, tf_idf_dict):
    best_resumes_method = {}

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_roberta, documents): "ROBERTA",
            executor.submit(process_tf_idf_key_word, documents, tokens, tf_idf_dict, top_n): "TF_IDF_KEY_WORD",
            executor.submit(process_tf_idf_fasttext, documents, tokens, tf_idf_dict, top_n): "TF_IDF_FASTTEXT",
            executor.submit(process_bert, documents, top_n): "BERT",

        }

        for future in as_completed(futures):
            method_name, result = future.result()
            if method_name == "ROBERTA":
                for k, v in result.items():
                    best_resumes_method[k] = v
            else:
                best_resumes_method[method_name] = result

    # print(len(best_resumes_method.keys()))
    return list(best_resumes_method.keys()), list(best_resumes_method.values())


def get_best_method(methodes, n_documents, top_n):
    scores = np.zeros((len(methodes.keys())+2, n_documents), dtype=int)

    for idx, file in enumerate(tqdm(glob.glob(os.path.join(os.getenv('TXT_FOLDER2'), "*.txt"))[0:n_documents])):
        with open(file, "r", encoding="utf-8") as f:
            doc = f.read().strip().lower()
            if len(doc) > 10:
                documents = np.array(sent_tokenize(doc))

                # Prétraitement
                tokens, tf_idf_dict = processor.preprocess(documents)
                methode_names, documents_resume = get_resume_by_method(
                    documents, top_n=top_n, tokens=tokens, tf_idf_dict=tf_idf_dict)

                P, R, F1 = get_bert_score(
                    documents_resume, [doc] * (len(methodes.keys())+2))
                F1 = F1.numpy()
                scores[:, idx] += (np.array(np.argsort(F1)[::-1]) + 1)

                # del documents, tokens, tf_idf_dict, documents_resume
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()  # Forcer le ramassage des ordures
                # processor_roberta.process_document(documents)

    return scores, scores.mean(axis=1), P, R, F1, methode_names



if __name__ == '__main__':
    # mp.set_start_method('spawn')
    scores, score_index, P, R, F1, methodes_name = get_best_method(
        methodes, 10, 2)
    print(methodes_name, scores, score_index, F1)
