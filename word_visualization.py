# import matplotlib as mp
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
# Télécharger les stopwords en français
nltk.download('stopwords')
nltk.download('punkt_tab')

# Fonction pour nettoyer et tokeniser le texte
# Fonction pour nettoyer et tokeniser le texte
def nettoyer_texte(texte):
    texte = texte.lower()  # Mettre en minuscule
    texte = texte.translate(str.maketrans('', '', string.punctuation))  # Supprimer la ponctuation
    mots = word_tokenize(texte)  # Tokenisation
    mots_filtres = [mot for mot in mots if mot not in stopwords.words('french')]  # Supprimer les stopwords
    return mots_filtres

# Dataset
# corpus = {
#     "doc1": ["Le traitement automatique du langage naturel est un domaine fascinant de l'intelligence artificielle.", 1],
#     "doc2": ["Les modèles de langage modernes utilisent des réseaux neuronaux pour analyser et générer du texte.", 1],
#     "doc3": ["mon chien a mangé du pain ce matin", 2]
# }
data = pd.read_csv("data/recup_pred_medias.tsv", sep="\t")
data = data[["text", "pred"]]
print(data)
data.set_index('pred',inplace=True)
# Regrouper les textes par cluster
corpus = data.to_dict()
clusters = {}
for doc, (texte, cluster) in corpus.items():
    if cluster not in clusters:
        clusters[cluster] = []
    clusters[cluster].append(texte)

# Générer et afficher un nuage de mots pour chaque cluster
# mp.use("GTK3Agg")
for cluster, textes in clusters.items():
    texte_complet = ' '.join(textes)
    mots_filtres = nettoyer_texte(texte_complet)
    frequences = Counter(mots_filtres)
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate_from_frequencies(frequences)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nuage de mots - Cluster {cluster}')
    plt.show()
    plt.savefig(f"visualizations/wordscloud{cluster}.jpg", bbox_inches="tight")
