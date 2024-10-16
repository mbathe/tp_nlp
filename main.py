from src.code.TFIDF.cleaning_paul import Processing

import os

current_working_directory = os.getcwd()

if __name__ == '__main__':
    preprocessing = Processing()
    my_preprocessed_corpus = preprocessing.process_corpus()
    # print(my_preprocessed_corpus.__getattribute__('tf_idf_dict'))
