# import matplotlib as mp
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import os
import numpy as np
from scipy.sparse import load_npz
import umap.umap_ as umap
import umap.plot as umap_plot 
from typing import List, Union
import plotly.graph_objects as go
import collections
import plotly.express as px

def main_scatterplot(**args):

    def build_path(**args):
        if args["origin_dataset"].startswith("event2018"):
            dataset = args["origin_dataset"].replace(".tsv", "")
        else:
            dataset = args["origin_dataset"].split("/")[-1].replace(".tsv", "")
        file_name = args.get("annotation", "vectors")
        for arg in ["text+", "hashtag_split", "svd", "tfidf_weights"]:
            if arg in args and args[arg]:
                file_name += "_" + arg
        if args["model"] == "sbert":
            sbert_model = args["sub_model"].replace("/", "-")
            file_name += "_" + sbert_model
        return os.path.join("data", dataset, args["model"], file_name)

    def load_matrix(**args):
        path = build_path(**args)
        print(path)
        for suffix in [".npy"]:
            if os.path.exists(path + suffix):
                return (
                    np.load(path + suffix)
                    if suffix == ".npy"
                    else load_npz(path + suffix)
                )
            
    def old(**args):
        # Télécharger les stopwords en français
        nltk.download("stopwords")
        nltk.download("punkt_tab")

        # Fonction de chatgpt pour nettoyer et tokeniser le texte
        def nettoyer_texte(texte):
            texte = texte.lower()  # Mettre en minuscule
            texte = texte.translate(
                str.maketrans("", "", string.punctuation)
            )  # Supprimer la ponctuation
            mots = word_tokenize(texte)  # Tokenisation
            mots_filtres = [
                mot for mot in mots if mot not in stopwords.words("french")
            ]  # Supprimer les stopwords
            return mots_filtres

        data = pd.read_csv("data/recup_pred_medias.tsv", sep="\t")
        data = data[["text", "pred"]]
        # mise en forme de dictionnaire contenant des listes[pred,text]
        corpus = data.transpose().to_dict()
        print(f"premiere clé du corpus :{corpus[1]}")
        clusters = {}
        for doc, values in corpus.items():
            if values["pred"] not in clusters:
                clusters[values["pred"]] = []
            clusters[values["pred"]].append(values["text"])

        print(f"{len(clusters)} clusters, first keys : {clusters[1]}")
        new_dict = dict()
        for cluster, textes in clusters.items():
            texte_complet = " ".join(textes)
            mots_filtres = nettoyer_texte(texte_complet)
            frequences = Counter(mots_filtres)
            new_dict[cluster] = frequences
        # topic_list = sorted(topics)
        # frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
        output = pd.DataFrame(new_dict)
        print(output)

    def main(**args):
        # old(**args)
        X = load_matrix(**args)
        if X is None:
            print("Matrix not found")
            return
        data = pd.read_csv("data/recup_pred_medias.tsv", sep="\t")
        data = data[["id", "pred"]]
        print(data)
        mapper = umap.UMAP().fit(X)
        # mapper = umap.UMAP().fit(fmnist.data) pour des grands jeux de données
        umap_plot.points(mapper, labels = data["pred"])
        plt.savefig("umap.png")

    main(**args)


main_scatterplot(
    origin_dataset="event2018_news.tsv",
    pred_dataset="recup_pred_medias.stv",
    model="sbert",
    sub_model="Lajavaness/sentence-camembert-large",
    annotation="annotated",
    hashtag_split=True
)
    # def useless() :
    # # Visualize with plotly
    # df = pd.DataFrame(
    #     {
    #         "x": embeddings[:, 0],
    #         "y": embeddings[:, 1],
    #         "Topic": topic_list,
    #         "Words": words,
    #         "Size": frequencies,
    #     }
    # )
    # return _plotly_topic_visualization(df, topic_list, title, width, height)


def _plotly_topic_visualization(df: pd.DataFrame, topic_list: List[str], title: str, width: int, height: int):
    """Create plotly-based visualization of topics with a slider for topic selection."""

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = ["red" if topic == topic_selected else "#B0BEC5" for topic in topic_list]
        return [{"marker.color": [marker_color]}]

    # Prepare figure range
    x_range = (
        df.x.min() - abs((df.x.min()) * 0.15),
        df.x.max() + abs((df.x.max()) * 0.15),
    )
    y_range = (
        df.y.min() - abs((df.y.min()) * 0.15),
        df.y.max() + abs((df.y.max()) * 0.15),
    )

    # Plot topics
    fig = px.scatter(
        df,
        x="x",
        y="y",
        size="Size",
        size_max=40,
        template="simple_white",
        labels={"x": "", "y": ""},
        hover_data={"Topic": True, "Words": True, "Size": True, "x": False, "y": False},
    )
    fig.update_traces(marker=dict(color="#B0BEC5", line=dict(width=2, color="DarkSlateGrey")))

    # Update hover order
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "<b>Topic %{customdata[0]}</b>",
                "%{customdata[1]}",
                "Size: %{customdata[2]}",
            ]
        )
    )

    # Create a slider for topic selection
    steps = [dict(label=f"Topic {topic}", method="update", args=get_color(topic)) for topic in topic_list]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    # Stylize layout
    fig.update_layout(
        title={
            "text": f"{title}",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": dict(size=22, color="Black"),
        },
        width=width,
        height=height,
        hoverlabel=dict(bgcolor="white", font_size=16, font_family="Rockwell"),
        xaxis={"visible": False},
        yaxis={"visible": False},
        sliders=sliders,
    )

    # Update axes ranges
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)

    # Add grid in a 'plus' shape
    fig.add_shape(
        type="line",
        x0=sum(x_range) / 2,
        y0=y_range[0],
        x1=sum(x_range) / 2,
        y1=y_range[1],
        line=dict(color="#CFD8DC", width=2),
    )
    fig.add_shape(
        type="line",
        x0=x_range[0],
        y0=sum(y_range) / 2,
        x1=x_range[1],
        y1=sum(y_range) / 2,
        line=dict(color="#9E9E9E", width=2),
    )
    fig.add_annotation(x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10)
    fig.add_annotation(y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10)
    fig.data = fig.data[::-1]

    return fig










# faire des nuages de mots sur (quelques) clusters
def main_wordcloud():
    # Télécharger les stopwords en français
    nltk.download("stopwords")
    nltk.download("punkt_tab")

    # Fonction de chatgpt pour nettoyer et tokeniser le texte
    def nettoyer_texte(texte):
        texte = texte.lower()  # Mettre en minuscule
        texte = texte.translate(
            str.maketrans("", "", string.punctuation)
        )  # Supprimer la ponctuation
        mots = word_tokenize(texte)  # Tokenisation
        mots_filtres = [
            mot for mot in mots if mot not in stopwords.words("french")
        ]  # Supprimer les stopwords
        return mots_filtres

    data = pd.read_csv("data/recup_pred_medias.tsv", sep="\t")
    data = data[["text", "pred"]]
    print(data)
    # mise en forme de dictionnaire contenant des listes[pred,text]
    corpus = data.transpose().to_dict()
    print(corpus[1])
    clusters = {}
    for doc, values in corpus.items():
        if values["pred"] not in clusters:
            clusters[values["pred"]] = []
        clusters[values["pred"]].append(values["text"])

    # Générer et afficher un nuage de mots pour chaque cluster
    # mp.use("GTK3Agg")
    for cluster, textes in clusters.items():
        texte_complet = " ".join(textes)
        mots_filtres = nettoyer_texte(texte_complet)
        frequences = Counter(mots_filtres)

        wordcloud = WordCloud(
            width=800, height=400, background_color="white", colormap="viridis"
        ).generate_from_frequencies(frequences)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuage de mots - Cluster {cluster}")
        plt.savefig(f"visualizations/wordscloud{cluster}.jpg", bbox_inches="tight")
