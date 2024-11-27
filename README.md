# Résumé Automatique et Extraction de Mots-Clés

Ce projet a été réalisé dans le cadre d'un cours de Traitement Automatique du Langage Naturel (NLP). Il se concentre sur l'extraction de mots-clés et la génération de résumés à partir de documents textuels en utilisant diverses approches, notamment TF-IDF, FastText, et des modèles avancés comme BERT et Roberta.

## Fonctionnalités
- **Extraction de mots-clés** :  
  - Approche basée sur les poids TF-IDF.  
  - Méthodes sémantiques exploitant les embeddings FastText et des regroupements sémantiques.  
  - Utilisation de graphes de mots pour modéliser les relations lexicales.  

- **Génération de résumés** :  
  - Identification de la phrase résumant au mieux le document grâce aux mots-clés.  
  - Approches sémantiques avec FastText et BERT pour aligner phrases et documents.  
  - Variantes avancées utilisant Roberta pour des performances accrues.  

---

## Prérequis

Avant d'exécuter le code, assurez-vous que les éléments suivants sont installés et configurés :  

- Python 3.11 ou supérieur.  
- **Poetry** pour la gestion des dépendances. [Installer Poetry](https://python-poetry.org/docs/#installation).  
- Modèle **FastText** pré-entraîné : [cc.en.300.bin](https://fasttext.cc/docs/en/crawl-vectors.html).  
- Modèle et ressources Spacy :  
  - Stop words anglais (`"english"`)  
  - Modèle **en_core_web_sm** ([Lien Spacy](https://spacy.io/models/en))  

---

## Installation

### Étape 1 : Clonez le dépôt
```bash
git clone https://github.com/mbathe/tp_nlp.git
cd tp_nlp
```


### Étape 2 : Installez les dépendances
poetry install

### Étape 3 : Téléchargez les ressources nécessaires
1. Téléchargez le modèle cc.en.300.bin de FastText à partir de ce lien :
https://fasttext.cc/docs/en/crawl-vectors.html.

Placez le fichier dans le répertoire racine du projet
2. Téléchargez les stop words anglais de Spacy :
```
 python -m spacy download en_core_web_sm
 python -m spacy download en
```
1. Télécharger le dataset 
Télécharger le https://drive.google.com/file/d/1YoM8A2X5spemIS2palnZTVKyQk7xFMJR/view?usp=sharing à partir de le lien drive  déjà regroupper et trier et placer le à l'emplacement  "p_nlp/src/docfile"
 

4. Le fichier notebook evaluate.ipynb vous permet de tester les nos différentes solutions, vous y trouverez les les résultats d'exécution de lagorithme sur le jeu de donnée

Les différentes approches proposées ont été testées sur un corpus portant sur l'éthique de l'intelligence artificielle. Les résultats obtenus montrent que :

La méthode basée sur TF-IDF est très efficace pour extraire des mots-clés pertinents et trouver les phrases résumant les documents.
L'utilisation des embeddings sémantiques via FastText et BERT améliore la qualité des résumés, notamment dans des contextes où des synonymes ou des reformulations apparaissent.
Les variantes avec Roberta ont également montré des performances très prometteuses pour cette tâche.

