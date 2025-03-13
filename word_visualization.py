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
from typing import List, Union, Mapping, Tuple
import plotly.graph_objects as go
import plotly.express as px
import collections
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator


"""
plot by using a umap reduction
"""


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
        umap_plot.points(mapper, labels=data["pred"])
        plt.savefig("umap.png")

    main(**args)


main_scatterplot(
    origin_dataset="event2018_news.tsv",
    pred_dataset="recup_pred_medias.stv",
    model="sbert",
    sub_model="Lajavaness/sentence-camembert-large",
    annotation="annotated",
    hashtag_split=True,
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


"""
create topic visualization by using plotly (only a generated code for now)
"""


def _plotly_topic_visualization(
    df: pd.DataFrame, topic_list: List[str], title: str, width: int, height: int
):
    """Create plotly-based visualization of topics with a slider for topic selection."""

    def get_color(topic_selected):
        if topic_selected == -1:
            marker_color = ["#B0BEC5" for _ in topic_list]
        else:
            marker_color = [
                "red" if topic == topic_selected else "#B0BEC5" for topic in topic_list
            ]
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
    fig.update_traces(
        marker=dict(color="#B0BEC5", line=dict(width=2, color="DarkSlateGrey"))
    )

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
    steps = [
        dict(label=f"Topic {topic}", method="update", args=get_color(topic))
        for topic in topic_list
    ]
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
    fig.add_annotation(
        x=x_range[0], y=sum(y_range) / 2, text="D1", showarrow=False, yshift=10
    )
    fig.add_annotation(
        y=y_range[1], x=sum(x_range) / 2, text="D2", showarrow=False, xshift=10
    )
    fig.data = fig.data[::-1]

    return fig


"""
generate a wordcloud visalizations for each event (not really viable bc of the nb of events, but still interesting)
"""


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


"""
not yet adapted :
re-using BERTopic code parts
"""


class BaseRepresentation(BaseEstimator):
    """The base representation model for fine-tuning topic representations."""

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topics.

        Each representation model that inherits this class will have
        its arguments (topic_model, documents, c_tf_idf, topics)
        automatically passed. Therefore, the representation model
        will only have access to the information about topics related
        to those arguments.

        Arguments:
            topic_model: The BERTopic model that is fitted until topic
                         representations are calculated.
            documents: A dataframe with columns "Document" and "Topic"
                       that contains all documents with each corresponding
                       topic.
            c_tf_idf: A c-TF-IDF representation that is typically
                      identical to `topic_model.c_tf_idf_` except for
                      dynamic, class-based, and hierarchical topic modeling
                      where it is calculated on a subset of the documents.
            topics: A dictionary with topic (key) and tuple of word and
                    weight (value) as calculated by c-TF-IDF. This is the
                    default topics that are returned if no representation
                    model is used.
        """
        return topic_model.topic_representations_


class topic_model:
    def __init__(self):
        self.topics_ = None
        self.topic_sizes_ = None
        self.topic_aspects_ = None
        self.custom_labels_ = None
        self._outliers = None
        self.c_tf_idf_ = None
        self.topic_embeddings_ = None
        self.vectorizer_model = CountVectorizer(ngram_range=self.n_gram_range)

    def fill_in(self, documents):
        self.topics_ = documents.Topic.astype(int).to_list()
        self.topic_sizes_ = collections.Counter(documents.Topic.values.tolist())

    def get_topic_freq():
        pass

    def get_topic(topic):
        pass

    @property
    def _outliers(self):
        """Some algorithms have outlier labels (-1) that can be tricky to work
        with if you are slicing data based on that labels. Therefore, we
        track if there are outlier labels and act accordingly when slicing.

        Returns:
            An integer indicating whether outliers are present in the topic model
        """
        return 1 if -1 in self.topic_sizes_ else 0

    # def _c_tf_idf(
    #     self,
    #     documents_per_topic: pd.DataFrame,
    #     fit: bool = True,
    #     partial_fit: bool = False,
    # ) -> Tuple[csr_matrix, List[str]]:
    #     documents = self._preprocess_text(documents_per_topic.Document.values)
    #     X = self.vectorizer_model.transform(documents)

    def _extract_words_per_topic(
        self,
        words: List[str],
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix = None,
        fine_tune_representation: bool = True,
        calculate_aspects: bool = False,
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Based on tf_idf scores per topic, extract the top n words per topic.

        If the top words per topic need to be extracted, then only the `words` parameter
        needs to be passed. If the top words per topic in a specific timestamp, then it
        is important to pass the timestamp-based c-TF-IDF matrix and its corresponding
        labels.

        Arguments:
            words: List of all words (sorted according to tf_idf matrix position)
            documents: DataFrame with documents and their topic IDs
            c_tf_idf: A c-TF-IDF matrix from which to calculate the top words
            fine_tune_representation: If True, the topic representation will be fine-tuned using representation models.
                                      If False, the topic representation will remain as the base c-TF-IDF representation.
            calculate_aspects: Whether to calculate additional topic aspects

        Returns:
            topics: The top words per topic
        """
        if c_tf_idf is None:
            print("need c_tf_idf")

        labels = sorted(list(documents.Topic.unique()))
        labels = [int(label) for label in labels]

        # Get at least the top 30 indices and values per row in a sparse c-TF-IDF matrix
        top_n_words = max(self.top_n_words, 30)
        indices = self._top_n_idx_sparse(c_tf_idf, top_n_words)
        scores = self._top_n_values_sparse(c_tf_idf, indices)
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)

        # Get top 30 words per topic based on c-TF-IDF score
        base_topics = {
            label: [
                (words[word_index], score)
                if word_index is not None and score > 0
                else ("", 0.00001)
                for word_index, score in zip(indices[index][::-1], scores[index][::-1])
            ]
            for index, label in enumerate(labels)
        }

        # Fine-tune the topic representations
        topics = base_topics.copy()
        if not self.representation_model or not fine_tune_representation:
            # Default representation: c_tf_idf + top_n_words
            topics = {
                label: values[: self.top_n_words] for label, values in topics.items()
            }
        elif fine_tune_representation and isinstance(self.representation_model, list):
            for tuner in self.representation_model:
                topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
        elif fine_tune_representation and isinstance(
            self.representation_model, BaseRepresentation
        ):
            topics = self.representation_model.extract_topics(
                self, documents, c_tf_idf, topics
            )
        elif fine_tune_representation and isinstance(self.representation_model, dict):
            if self.representation_model.get("Main"):
                main_model = self.representation_model["Main"]
                if isinstance(main_model, BaseRepresentation):
                    topics = main_model.extract_topics(
                        self, documents, c_tf_idf, topics
                    )
                elif isinstance(main_model, list):
                    for tuner in main_model:
                        topics = tuner.extract_topics(self, documents, c_tf_idf, topics)
                else:
                    raise TypeError(
                        f"unsupported type {type(main_model).__name__} for representation_model['Main']"
                    )
            else:
                # Default representation: c_tf_idf + top_n_words
                topics = {
                    label: values[: self.top_n_words]
                    for label, values in topics.items()
                }
        else:
            raise TypeError(
                f"unsupported type {type(self.representation_model).__name__} for representation_model"
            )

        # Extract additional topic aspects
        if calculate_aspects and isinstance(self.representation_model, dict):
            for aspect, aspect_model in self.representation_model.items():
                if aspect != "Main":
                    aspects = base_topics.copy()
                    if not aspect_model:
                        # Default representation: c_tf_idf + top_n_words
                        aspects = {
                            label: values[: self.top_n_words]
                            for label, values in aspects.items()
                        }
                    if isinstance(aspect_model, list):
                        for tuner in aspect_model:
                            aspects = tuner.extract_topics(
                                self, documents, c_tf_idf, aspects
                            )
                    elif isinstance(aspect_model, BaseRepresentation):
                        aspects = aspect_model.extract_topics(
                            self, documents, c_tf_idf, aspects
                        )
                    else:
                        raise TypeError(
                            f"unsupported type {type(aspect_model).__name__} for representation_model[{repr(aspect)}]"
                        )
                    self.topic_aspects_[aspect] = aspects

        return topics


def select_topic_representation():
    pass


def visualize_topics(
    topic_model,
    topics: List[int] = None,
    top_n_topics: int = None,
    use_ctfidf: bool = False,
    custom_labels: Union[bool, str] = False,
    title: str = "<b>Intertopic Distance Map</b>",
    width: int = 650,
    height: int = 650,
) -> go.Figure:
    """Visualize topics, their sizes, and their corresponding words.

    This visualization is highly inspired by LDAvis, a great visualization
    technique typically reserved for LDA.

    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        use_ctfidf: Whether to use c-TF-IDF representations instead of the embeddings from the embedding model.
        custom_labels: If bool, whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
                       If `str`, it uses labels from other aspects, e.g., "Aspect1".
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
    """
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Extract topic words and their frequencies
    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes_[topic] for topic in topic_list]
    if isinstance(custom_labels, str):
        words = [
            [[str(topic), None]] + topic_model.topic_aspects_[custom_labels][topic]
            for topic in topic_list
        ]
        words = ["_".join([label[0] for label in labels[:4]]) for labels in words]
        words = [label if len(label) < 30 else label[:27] + "..." for label in words]
    elif custom_labels and topic_model.custom_labels_ is not None:
        words = [
            topic_model.custom_labels_[topic + topic_model._outliers]
            for topic in topic_list
        ]
    else:
        words = [
            " | ".join([word[0] for word in topic_model.get_topic(topic)[:5]])
            for topic in topic_list
        ]

    # Embed c-TF-IDF into 2D
    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])

    embeddings, c_tfidf_used = select_topic_representation(
        topic_model.c_tf_idf_,
        topic_model.topic_embeddings_,
        use_ctfidf=use_ctfidf,
        output_ndarray=True,
    )
    embeddings = embeddings[indices]

    if c_tfidf_used:
        embeddings = MinMaxScaler().fit_transform(embeddings)
        embeddings = umap.UMAP(
            n_neighbors=2, n_components=2, metric="hellinger", random_state=42
        ).fit_transform(embeddings)
    else:
        embeddings = umap.UMAP(
            n_neighbors=2, n_components=2, metric="cosine", random_state=42
        ).fit_transform(embeddings)

    # Visualize with plotly
    df = pd.DataFrame(
        {
            "x": embeddings[:, 0],
            "y": embeddings[:, 1],
            "Topic": topic_list,
            "Words": words,
            "Size": frequencies,
        }
    )
    return _plotly_topic_visualization(df, topic_list, title, width, height)
