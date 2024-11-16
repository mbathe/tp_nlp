import os
import glob
import shutil
from langdetect import detect, DetectorFactory
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
# Pour rendre la détection reproductible
DetectorFactory.seed = 0

# Dictionnaire de correspondance des codes langues vers les noms complets
lang_names = {
    "en": "anglais",
    "fr": "français",
    "de": "allemand",
    "es": "espagnol",
    "nl": "néerlandais",
    "cs": "tchèque",
    "pl": "polonais",
    "pt": "portugais",
    "zh-cn": "chinois simplifié",
    "ca": "catalan",
    "sv": "suédois",
    "id": "indonésien",
    "it": "italien",
    "cy": "gallois"
}

# Initialiser le compteur de langues et une liste pour les fichiers non-anglais
lang_count = defaultdict(int)
non_english_files = []
english_files = []

# Dossier contenant les fichiers texte
txt_folder = os.getenv('TXT_FOLDER')
output_folder = os.getenv('TXT_FOLDER2')

# Créer le nouveau dossier txts_2 s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Vérifier si le dossier source existe
if not os.path.exists(txt_folder):
    # print(f"Le dossier {txt_folder} n'existe pas.")
    exit()

# Parcourir chaque fichier dans le dossier
for filepath in glob.glob(os.path.join(txt_folder, '*.txt')):
    try:
        # Lire le contenu du fichier
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().strip()  # Retire les espaces inutiles
            # Vérifier que le texte n'est pas vide ou trop court
            if len(text) < 20:  # Si le texte contient moins de 20 caractères, ignorer
                # print(
                    f"Le fichier {filepath} est trop court ou vide, il est ignoré.")
                continue
            # Détecter la langue du texte
            lang = detect(text)
            # Incrémenter le compteur pour cette langue
            lang_count[lang] += 1
            # Si la langue détectée est l'anglais, copier le fichier dans txts_2
            if lang == "en":
                english_files.append(filepath)
                # Copier le fichier en anglais dans txts_2
                shutil.copy(filepath, output_folder)
            else:
                non_english_files.append(filepath)
    except Exception as e:
        #print(f"Erreur lors du traitement de {filepath}: {e}")

# Afficher le résultat avec les noms complets des langues
#print("Nombre de fichiers par langue détectée :")
for lang, count in lang_count.items():
    # Utiliser le nom complet ou le code si inconnu
    lang_full = lang_names.get(lang, lang)
    #print(f"{lang_full}: {count} fichiers")

# Afficher les fichiers qui ne sont pas en anglais
if non_english_files:
    #print("\nFichiers non anglais :")
    for filename in non_english_files:
        #print(filename)
else:
    #print("\nTous les fichiers sont en anglais.")

# Afficher le nombre de fichiers copiés dans txts_2
#print(f"\n{len(english_files)} fichiers anglais ont été copiés dans {output_folder}.")
