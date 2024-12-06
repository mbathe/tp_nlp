import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_excel('mots_importants_par_texte.xlsx', engine='openpyxl')

word_counts = Counter(df['Mot'])

sorted_word_counts = word_counts.most_common()

top_10_words = sorted_word_counts[:10]
words, counts = zip(*top_10_words)

plt.figure(figsize=(10, 6))
plt.bar(words, counts, color='skyblue')
plt.title('10 mots les plus fréquents à travers tous les textes')
plt.xlabel('Mots')
plt.ylabel('Fréquence')
plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

algorithms = df['Algorithme'].unique()

for algo in algorithms:

    algo_df = df[df['Algorithme'] == algo]

    word_counts = Counter(algo_df['Mot'])
    sorted_word_counts = word_counts.most_common()

    top_10_words = sorted_word_counts[:10]
    words, counts = zip(*top_10_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue')
    plt.title(f'10 mots les plus fréquents pour l\'algorithme {algo}')
    plt.xlabel('Mots')
    plt.ylabel('Fréquence')
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()
