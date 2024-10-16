from src.code.TFIDF.cleaning_paul import Processing

import os

current_working_directory = os.getcwd()

if __name__ == '__main__':
    preprocessing = Processing(
        log_file=current_working_directory+"\\src\\code\\TFIDF\\logs\\", corpus_dir=current_working_directory+"\\src\\docfile\\processed\\", corpus_file=current_working_directory+"\\src\\docfile\\corpus.txt")
    print('lezgo')
    preprocessing.process_corpus()
