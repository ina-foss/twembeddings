import pickle
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from scipy import sparse
import numpy as np
import os.path
import re
from unidecode import unidecode
from tqdm import tqdm
from .stop_words import STOP_WORDS_FR, STOP_WORDS_EN
import warnings

__all__ = ['W2V', 'TfIdf', 'BERT', 'SBERT', 'Elmo', 'USE', 'DenseNetLayer', 'ResNetLayer', 'SIFT']
TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

class W2V:
    def __init__(self, model, lang="fr"):
        from gensim.models import KeyedVectors
        self.vocab = {}
        self.name = "w2v"
        self.lang = lang

        if lang == "fr":
            if model=="w2v_twitter_fr":
                self.n_features = 300
                path = "/home/bmazoyer/Dev/Twitter_OTM/clustering/data/twitter-10-300.w2v.model"
                self.wv = KeyedVectors.load_word2vec_format(path, binary=True)
                self.vocab = self.wv.vocab

            elif model=="w2v_afp_fr":
                self.n_features = 300
                path = "/home/bmazoyer/Dev/Twitter_OTM/clustering/data/afp-10-300.w2v.model"
                self.wv = KeyedVectors.load_word2vec_format(path, binary=True)
                self.vocab = self.wv.vocab
            else:
                raise NameError("Available w2v models for French are w2v_twitter_fr and w2v_afp_fr")
        elif lang == "en":
            if model == "w2v_twitter_en":
                self.n_features = 400
                from word2vecReader import Word2VecVariant
                path = "/home/bmazoyer/Dev/Twitter_OTM/clustering/data/mcminn/word2vec_twitter_model.bin"
                model = Word2VecVariant.load_word2vec_format(path, binary=True)
                self.vocab = model.vocab
                self.wv = model

            elif model == "w2v_gnews_en":
                self.n_features = 300
                import gensim.downloader
                path = os.path.join(gensim.downloader.base_dir, "word2vec-google-news-300/word2vec-google-news-300.gz")
                if os.path.exists(path):
                    self.wv = KeyedVectors.load_word2vec_format(path, binary=True)
                else:
                    self.wv = gensim.downloader.load("word2vec-google-news-300")
                self.vocab = self.wv.vocab
            else:
                raise NameError("Available w2v models for English are w2v_twitter_fr and w2v_afp_fr")

    def preprocess(self, data):
        text = data.text.str.findall(TOKEN_PATTERN)
        return text

    def compute_vectors(self, data):
        logging.info("compute vectors")
        text = self.preprocess(data)
        vectors = np.zeros((len(text), self.n_features))
        for idx, sentence in text.iteritems():

            vectors[idx] = np.array(
                [self.wv[w] if w in self.vocab else np.zeros(self.n_features) for w in sentence]
            ).mean(axis=0)

            if np.all(np.isnan(vectors[idx])):
                vectors[idx] = np.zeros(self.n_features)

        del self.wv
        return vectors

    def compute_weighted_vectors(self, data, lang):
        tfidf = TfIdf().load_history(lang)
        logging.info("compute vectors")
        text = self.preprocess(data)
        vectors = np.zeros((len(text), self.n_features))
        idf = np.log((tfidf.n_samples + 1) / (tfidf.df + 1)) + 1
        idf_dict = {k: v for k, v in zip(tfidf.features_names, idf)}
        for idx, sentence in text.iteritems():
            vectors[idx] = np.array(
                [idf_dict.get(unidecode(w.lower()), 0)*self.wv[w] if w in self.vocab else np.zeros(self.n_features) for w in sentence]
            ).mean(axis=0)

            if np.all(np.isnan(vectors[idx])):
                vectors[idx] = np.zeros(self.n_features)

        del self.wv
        return vectors


class TfIdf:
    def __init__(self, lang="fr", binary=True, tokenizer="sklearn", no_pandas=False):
        self.df = np.array([])
        self.features_names = []
        self.n_samples = 0
        self.name = "tfidf"
        self.binary = binary
        self.tokenizer = self.custom_tokenizer if tokenizer=="fog" else None
        self.no_pandas = no_pandas
        if lang == "fr":
            self.stop_words = STOP_WORDS_FR
        elif lang == "en":
            self.stop_words = STOP_WORDS_EN

    def custom_tokenizer(self, document):
        from fog.tokenizers.words import WordTokenizer
        tokenizer = WordTokenizer(
            keep=['word', 'mention'],
            lower=True,
            unidecode=True,
            split_hashtags=True,
            stoplist=self.stop_words + [t + "'" for t in self.stop_words] + [t + "’" for t in self.stop_words],
            reduce_words=True,
            decode_html_entities=True
        )
        return list(token for _, token in tokenizer(document))

    def load_history(self, lang):
        if lang == "fr":
            dataset = "event2018"
        else:
            dataset = "event2012"
        for attr in ["df", "features_names", "n_samples"]:
            with open("twembeddings/models/" + dataset + "_" + attr, "rb") as f:
                setattr(self, attr, pickle.load(f))
        return self

    def save(self, dataset):
        dataset = dataset.split("/")[-1].replace(".tsv", "")
        for attr in ["df", "features_names", "n_samples"]:
            with open("twembeddings/models/" + dataset + "_" + attr, "wb") as f:
                pickle.dump(getattr(self, attr), f)

    def get_new_features(self, data):
        features_set = set(self.features_names)
        fit_model = CountVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)
        # see https://towardsdatascience.com/hacking-scikit-learns-vectorizers-9ef26a7170af for custom analyzr/tokenizr
        if self.no_pandas:
            fit_model.fit(data)
        else:
            fit_model.fit(data["text"].tolist())
        for term in fit_model.get_feature_names():
            if term not in features_set:
                self.features_names.append(term)

    def build_count_vectors(self, data):
        # sort words following features_name order, absent words will be counted as 0
        count_model = CountVectorizer(binary=self.binary, vocabulary=self.features_names, tokenizer=self.tokenizer)
        if self.no_pandas:
            return count_model.transform(data)
        return count_model.transform(data["text"].tolist())

    def compute_df(self, count_vectors):
        # add zeros to the end of the stored df vector
        zeros = np.zeros(count_vectors.shape[1] - len(self.df), dtype=self.df.dtype)
        df = np.append(self.df, zeros)
        # compute new df array
        # np.bincount counts each time an index is present in count_vectors.indices
        # however it does not count "zero" for absent words
        # therefore we artificially add all indices: np.arange(count_vectors.shape[1])
        # and then substract 1 to all indices in the total score
        indices = np.hstack((count_vectors.indices, np.arange(count_vectors.shape[1])))
        df = df + np.bincount(indices) - 1
        return df

    def add_new_samples(self, data):
        self.get_new_features(data)
        count_vectors = self.build_count_vectors(data)
        self.df = self.compute_df(count_vectors)
        # logging.info("Count matrix shape: {}".format(count_vectors.shape))
        return count_vectors

    def compute_vectors(self, count_vectors, min_df, svd=False, n_components=0):
        if min_df > 0:
            mask = self.df > min_df
            df = self.df[mask]
            count_vectors = count_vectors[:,mask]
        else:
            df = self.df
        self.n_samples += count_vectors.shape[0]
        # logging.info("Min_df reduces nb of features, new count matrix shape: {}".format(
        #     count_vectors.shape)
        # )
        # compute smoothed idf
        idf = np.log((self.n_samples + 1) / (df + 1)) + 1
        transformer = TfidfTransformer()
        transformer._idf_diag = sparse.diags(idf, offsets=0, shape=(len(df), len(df)), format="csr", dtype=df.dtype)
        X = transformer.transform(count_vectors)
        # equivalent to:
        # X = normalize(X * transformer._idf_diag, norm='l2', copy=False)
        if svd:
            logging.info("Performing dimensionality reduction using LSA")
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            X = lsa.fit_transform(X)
            logging.info("New shape: {}".format(X.shape))

        return X


class BERT:

    def __init__(self):
        from bert_serving.client import BertClient
        self.name = "BERT"
        self.bc = BertClient()

    def compute_vectors(self, data):
        data["text"] = data.text.str.slice(0, 500)
        vectors = self.bc.encode(data.text.tolist())
        return vectors


class SBERT:

    def __init__(self, lang="fr"):
        from sentence_transformers import SentenceTransformer
        self.name = "SBERT"
        if lang == "fr":
            self.model = SentenceTransformer(
                "/home/bmazoyer/Dev/pytorch_bert/output/sts_fr_long_multilingual_bert-2019-10-01_15-07-03")
        elif lang == "en":
            self.model = SentenceTransformer(
                "bert-large-nli-stsb-mean-tokens"
            )

    def compute_vectors(self, data):
        data["text"] = data.text.str.slice(0, 500)
        vectors = np.array(self.model.encode(data.text.tolist()))
        return vectors


class Elmo:

    def __init__(self, lang="fr"):
        self.name = "elmo"
        if lang == "fr":
            from elmoformanylangs import Embedder
            self.e = Embedder('/home/bmazoyer/Dev/ELMoForManyLangs/150', batch_size=32)
            self.vectors = None
        elif lang == "en":
            import tensorflow as tf
            import tensorflow_hub as hub
            self.embed = hub.Module("https://tfhub.dev/google/elmo/2")
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            self.session.run(tf.tables_initializer())
        self.lang = lang

    def populate_array(self, data):
        logging.info(data.name)
        self.vectors[data.name] = np.mean(np.array(self.e.sents2elmo([data.text.split()]))[0], axis=0)

    def compute_vectors(self, data):
        n = data.shape[0]
        self.vectors = np.zeros((n, 1024))
        if self.lang == "fr":
            data.apply(self.populate_array, axis=1)
            return self.vectors
        elif self.lang == "en":
            batch_size = 64
            for i in tqdm(range(0, n, batch_size)):
                self.vectors[i:min(n, i + batch_size)] = self.session.run(
                    self.embed(data.text[i:min(n, i + batch_size)].tolist(), signature="default", as_dict=True)[
                        "default"]
                )
            return self.vectors


class USE:

    def __init__(self, lang="fr"):
        import tensorflow as tf
        import tensorflow_hub as hub
        import tf_sentencepiece
        self.name = "UniversalSentenceEncoder"
        # Graph set up.
        g = tf.Graph()
        with g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            if lang == "fr":
                logging.info(lang)
                embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
            elif lang == "en":
                logging.info(lang)
                embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
            self.embedded_text = embed(self.text_input)
            init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Initialize session.
        self.session = tf.Session(graph=g)
        self.session.run(init_op)

    def compute_vectors(self, data):
        batch_size = 64
        n = data.shape[0]
        vectors = np.zeros([n, 512])
        for i in tqdm(range(0, n, batch_size)):
            vectors[i:min(n, i+batch_size)] = self.session.run(
                self.embedded_text,
                feed_dict={self.text_input: data.text[i:min(n, i+batch_size)].tolist()})
        return vectors


class DenseNetLayer:

    def __init__(self):
        from keras.models import Model
        from keras.applications.densenet import DenseNet121

        base_model = DenseNet121(weights="imagenet", include_top=True)
        self.featurizer = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        self.name = "DenseNet"

    def compute_vectors(self, image_path, batch_size=64, weight=1):
        from keras.preprocessing import image
        from keras.applications.densenet import preprocess_input
        batch_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
        flow = batch_gen.flow_from_directory(image_path,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             target_size=(224, 224),
                                             class_mode=None)
        X = self.featurizer.predict_generator(flow, steps=len(flow.filenames)/batch_size, verbose=1)

        if weight != 1:
            X = X*weight

        return X


class ResNetLayer:

    def __init__(self, ):
        from keras.models import Model
        from keras.applications.resnet50 import ResNet50

        base_model = ResNet50(weights='imagenet', include_top=True)
        self.featurizer = Model(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)
        self.name = "ResNet"

    def compute_vectors(self, image_path, batch_size=64, weight=1):
        from keras.preprocessing import image
        from keras.applications.resnet50 import preprocess_input
        batch_gen = image.ImageDataGenerator(preprocessing_function=preprocess_input)
        flow = batch_gen.flow_from_directory(image_path,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             target_size=(224, 224),
                                             class_mode=None)
        X = self.featurizer.predict_generator(flow, steps=len(flow.filenames)/batch_size, verbose=1)

        if weight != 1:
            X = X*weight

        return X

class SIFT:
    def __init__(self, ):
        self.name = "sift"

    def compute_vectors(self, vectors_path):
        X = np.load(vectors_path)
        diag = np.diag(X)
        X = 1 / (1 + (X - np.diag(diag)))
        return X
