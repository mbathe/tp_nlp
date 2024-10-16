### Commande pour la règle `download` ###

15 : log_fp = open(LOG_FILE, "w", encoding="utf8")
37 : user_agents = [x.strip() for x in open(UA_FILE, encoding="utf8").readlines()]

python dl_docs.py

# Commande pour la règle `parse` (après téléchargement)

python parse_docs.py

### Commande pour la règle `preprocess` (après parsing) ###

python preprocess.py

13 : nltk.download('stopwords')
24 : name = filename.split('\\')[1].split('.')[0]
25 : with open(filename, encoding="utf8") as f:
62 : f = open(f"{OUT_FOLDER}/{i}.txt", "w", encoding="utf8")

### Commande pour la règle `corpus` (après préprocessing) ###

29 : keywords = json.load(open(args.themes, encoding="utf8"))
59 : f = open(fname, "r", encoding="utf8")
107 : file = open(f"{t.strip('*')}/{i}.txt", "w", encoding="utf8")

python create_corpus.py -d preprocessed/ -t themes.json
