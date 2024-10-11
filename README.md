# mapaie


## Creating a virtual environment

```
pip install virtualenv
virtualenv venv
```

## Getting started

Clone the repository, and make sure the python package virtualenv is installed.


```
cd mapaie
source venv/bin/activate
pip install -r requirements.txt
```

Snakemake should be installed on the side.

You can then run `snakemake -c4` to download PDF files and extract their contents. PDF files are stored in `./pdfs`, and textual contents in ` ./txts/`.

# 11/10/2024

Mot clef et résumé d'un texte
Plusieurs textes .txt en anglais, qui traitent de l'éthique de l'IA.

---> collecter les pdf/html ok
--> Parser ok
--> CLeaning ok
--> Tf Idf sur chaque document pour trouver les mots clefs
