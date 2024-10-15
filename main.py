from tp_nlp.src.code.CreateCorpus.cleaning_paul import Processing

if __name__ == '__main__':
    preprocessing = Processing(
        log_file="", corpus_dir="processed", corpus_file="corpus.txt")
    preprocessing.process_corpus()
