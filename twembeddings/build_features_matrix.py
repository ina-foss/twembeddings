# -*- coding: utf-8 -*-
import argparse
import logging
import pandas as pd
from .embeddings import TfIdf, W2V, BERT, SBERT, Elmo, ResNetLayer, DenseNetLayer, USE
from .embeddings import TOKEN_PATTERN
from .stop_words import STOP_WORDS_FR, STOP_WORDS_EN
from scipy.sparse import issparse, save_npz, load_npz
import numpy as np
import os
import re
import csv
import tensorflow_hub as hub
from unidecode import unidecode
from datetime import datetime, timedelta
from collections import deque, defaultdict
import math

__all__ = ['build_matrix', 'load_dataset', 'load_matrix', 'save_tokens_JLH']

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"
STANDARD_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ['tfidf_all_tweets', 'tfidf_dataset', 'w2v_afp_fr', 'w2v_gnews_en', 'w2v_twitter_fr',
                   "w2v_twitter_en", "elmo", "bert", "bert_tweets", "sbert", "sbert_sts", "sbert_stsshort",
                   "sbert_tweets_sts", "sbert_nli_sts", "sbert_tweets_sts_long", "use_multilingual", "use"]
image_embeddings = ["resnet", "densenet"]


def strp_date_created_at(created_at):
    if "+0000" in created_at:
        return datetime.strptime(created_at, TWITTER_DATE_FORMAT)
    return datetime.strptime(created_at, STANDARD_DATE_FORMAT)


def find_date_created_at(created_at):
    d = strp_date_created_at(created_at)
    return d.strftime("%Y%m%d"), d.strftime("%H:%M:%S")


def remove_repeted_characters(expr):
    #limit number of repeted letters to 3. For example loooool --> loool
    string_not_repeted = ""
    for item in re.findall(r"((.)\2*)", expr):
        if len(item[0]) <= 3:
            string_not_repeted += item[0]
        else:
            string_not_repeted += item[0][:3]
    return string_not_repeted


def camel_case_split(expr):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', expr)
    return " ".join([m.group(0) for m in matches])


def format_text(text, **format):
    # remove urls
    text = re.sub(r"http\S+", '', text, flags=re.MULTILINE)
    if format["remove_mentions"]:
        text = re.sub(r"@\S+", '', text, flags=re.MULTILINE)
    # translate to equivalent ascii characters
    if format["unidecode"]:
        text = unidecode(text)

    new_text = []
    for word in re.split(r"[' ]", text):
        # remove numbers longer than 4 digits
        if len(word) < 5 or not word.isdigit():
            if word.startswith("#") and format["hashtag_split"]:
                new_text.append(camel_case_split(word[1:]))
            else:
                new_text.append(word)
    text = remove_repeted_characters(" ".join(new_text))
    if format["lower"]:
        text = text.lower()
    return text


def build_path(**args):
    if args["dataset"].startswith("event2018"):
        dataset = args["dataset"]
    else:
        dataset = args["dataset"].split("/")[-1].replace(".tsv", "")

    file_name = args.get("annotation", "vectors")
    for arg in ["text+", "hashtag_split", "svd", "tfidf_weights"]:
        if arg in args and args[arg]:
            file_name += "_" + arg
    if args["model"] == "sbert":
        sbert_model = args["sub_model"].replace("/","-")
        file_name += "_" + sbert_model
    return os.path.join("data", dataset, args["model"], file_name)


def save_matrix(X, **args):
    path = build_path(**args)
    os.makedirs(os.path.join(*path.split("/")[:-1]), exist_ok=True)
    if issparse(X):
        save_npz(path, X)
    else:
        np.save(path, X)


def apply_mask(path, suffix, args, column):
    X = np.load(path + suffix) if suffix == ".npy" else load_npz(path + suffix)
    data = load_dataset(args["dataset"], args["annotation"], args["text+"])
    mask = data[column].notna()
    return X[mask]


def load_matrix(**args):
    path = build_path(**args)
    for suffix in [".npy"]:
        if os.path.exists(path + suffix):
            return np.load(path + suffix) if suffix == ".npy" else load_npz(path + suffix)

    if args["dataset"] == "event2018":
        if args["annotation"] == "annotated":
            args1 = args.copy()
            args1["annotation"] = "examined"
            path = build_path(**args1)
            for suffix in [".npy"]:
                if os.path.exists(path + suffix):
                    return apply_mask(path, suffix, args1, "label")
    elif args["dataset"] == "event2018_image":
        args1 = args.copy()
        args1["dataset"] = "event2018"
        path = build_path(**args1)
        for suffix in [".npy"]:
            if os.path.exists(path + suffix):
                return apply_mask(path, suffix, args1, "image")
    elif args["dataset"] == "event2018_image" and args["annotation"] == "annotated":
        args1 = args.copy()
        args1["annotation"] = "examined"
        args1["dataset"] = "event2018_image"
        path = build_path(**args1)
        if os.path.exists(path + suffix):
            return apply_mask(path, suffix, args1, "label")


def load_dataset(dataset, annotation, text=False):
    data = pd.read_csv(dataset,
                       sep="\t",
                       quoting=csv.QUOTE_ALL,
                       dtype={"id": str, "label": float, "created_at": str, "text": str}
                       )
    data.text = data.text.fillna("")
    if annotation == "annotated" and "label" in data.columns:
        data = data[data.label.notna()]
    elif annotation == "examined" and "label" in data.columns:
        data = data[data.event.notna()]
    if dataset == "data/event2018_image":
        data = data[data.image.notna()]

    if text == "text+" and "text+quote+reply" in data.columns:
        data = data.rename(columns={"text": "text_not_formated", "text+quote+reply": "text"})
    data["date"], data["time"] = zip(*data["created_at"].apply(find_date_created_at))
    return data.drop_duplicates("id").sort_values("id").reset_index(drop=True)


def save_tokens_JLH(inpath,
                    outpath,
                    window_size=12,
                    sep=",",
                    hashtag_split=True,
                    remove_mentions=False,
                    unidecode=True,
                    lower=True
                    ):
    window = deque()
    index = defaultdict(lambda: {"count": 0, "window_count": 0, "percent_max": 0})
    stop_words = STOP_WORDS_FR + STOP_WORDS_EN
    doc_count = 0
    if type(inpath) != list:
        inpath = [inpath]
    for filepath in inpath:
        logging.info(filepath)
        with open(filepath, "r") as f:
            reader = csv.reader(f, delimiter=sep)
            headers = next(reader)
            positions = {}
            for enum, h in enumerate(headers):
                positions[h] = enum
            for row in reader:
                doc_count += 1
                date = strp_date_created_at(row[positions["created_at"]])
                text = format_text(row[positions["text"]],
                                   remove_mentions=remove_mentions,
                                   unidecode=unidecode,
                                   lower=lower,
                                   hashtag_split=hashtag_split
                                   )
                tokens = [t for t in re.findall(TOKEN_PATTERN, text) if t not in stop_words]

                window.append((date, tokens))

                for t in tokens:
                    counts = index[t]
                    counts["count"] += 1
                    counts["window_count"] += 1
                    if counts["count"] != counts["window_count"]:
                        percent = counts["window_count"]/len(window)
                        counts["percent_max"] = max(percent, counts["percent_max"]) # type: ignore
                    if counts["percent_max"] == 1:
                        print(t, len(window), counts)

                if (window[-1][0] - window[0][0]).seconds / 3600 >= window_size:
                    first_element = window.popleft()
                    tokens = first_element[1]
                    for t in tokens:
                        counts = index[t]
                        counts["window_count"] -= 1


    sorted_count = sorted(index, key=lambda x:index[x]["count"], reverse=True)
    with open(outpath, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "df", "idf", "jlh_max"])
        for t in sorted_count:
            counts = index[t]
            df = counts["count"]
            idf = math.log((doc_count + 1) / (df + 1)) + 1
            # jlh = (pfore - pback)pfore/pback if pfore - pback > 0 , else 0
            percent_background = df/doc_count
            percent_foreground = counts["percent_max"]
            difference = percent_foreground - percent_background
            if difference > 0:
                jlh_max = difference*percent_foreground/percent_background
            else:
                jlh_max = 0
            writer.writerow([t, df, idf, jlh_max])
    return index


def build_matrix(**args):
    X = load_matrix(**args)
    if args["model"] in text_embeddings:
        data = load_dataset(args["dataset"], args["annotation"], args["text+"])
    if X is not None:
        logging.info("Matrix already stored")
        return X, data

    if args["model"].startswith("tfidf"):
        vectorizer = TfIdf(lang=args["lang"], binary=args["binary"], tokenizer="sklearn")
        if args["model"].endswith("all_tweets"):
            vectorizer.load_history(args["lang"])
        data.text = data.text.apply(format_text,
                                    remove_mentions=args["remove_mentions"],
                                    unidecode=True,
                                    lower=True,
                                    hashtag_split=args["hashtag_split"]
                                    )
        count_matrix = vectorizer.add_new_samples(data)
        X = vectorizer.compute_vectors(count_matrix, min_df=10, svd=args["svd"], n_components=100)

    elif args["model"].startswith("w2v"):
        vectorizer = W2V(args["model"], lang=args["lang"])
        if args["lang"] == "fr":
            data.text = data.text.apply(format_text,
                                        remove_mentions=args["remove_mentions"],
                                        unidecode=True,
                                        lower=True,
                                        hashtag_split=args["hashtag_split"]
                                        )
        elif args["model"] == "w2v_twitter_en":
            data.text = data.text.apply(format_text,
                                        remove_mentions=False,
                                        unidecode=False,
                                        lower=False,
                                        hashtag_split=args["hashtag_split"]
                                        )
        elif args["model"] == "w2v_gnews_en":
            data.text = data.text.apply(format_text,
                                        remove_mentions=args["remove_mentions"],
                                        unidecode=False,
                                        lower=False,
                                        hashtag_split=args["hashtag_split"]
                                        )
        if args["tfidf_weights"]:
            X = vectorizer.compute_weighted_vectors(data, args["lang"])
        else:
            X = vectorizer.compute_vectors(data)

    elif args["model"] == "elmo":
        data.text = data.text.apply(format_text,
                                    remove_mentions=args["remove_mentions"],
                                    unidecode=False,
                                    lower=False,
                                    hashtag_split=True
                                    )
        vectorizer = Elmo(lang=args["lang"])
        X = vectorizer.compute_vectors(data)

    elif args["model"].startswith("bert"):
        data.text = data.text.apply(format_text,
                                    remove_mentions=args["remove_mentions"],
                                    unidecode=False,
                                    lower=False,
                                    hashtag_split=True
                                    )
        vectorizer = BERT()
        X = vectorizer.compute_vectors(data)

    elif args["model"].startswith("sbert"):
        data.text = data.text.apply(format_text,
                                    remove_mentions=args["remove_mentions"],
                                    unidecode=False,
                                    lower=False,
                                    hashtag_split=True
                                    )

        vectorizer = SBERT(sbert_model=args["sub_model"])
        X = vectorizer.compute_vectors(data)

    elif args["model"].startswith("use"):
        data.text = data.text.apply(format_text,
                                    remove_mentions=args["remove_mentions"],
                                    unidecode=False,
                                    lower=False,
                                    hashtag_split=True
                                    )

        # todo: prevent warning message if no cuda with os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if args["lang"] == "en":
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        else:
            embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

        vectorizer = USE(embed) # type: ignore
        X = vectorizer.compute_vectors(data)

    elif args["model"] == "resnet":
        vectorizer = ResNetLayer()
        X = vectorizer.compute_vectors("data/images/event2018_image/")

    elif args["model"] == "densenet":
        vectorizer = DenseNetLayer()
        X = vectorizer.compute_vectors("data/images/event2018_image/")

    if args["save"]:
        save_matrix(X, **args)

    return X, data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--dataset',
                        required=False,
                        default="event2018",
                        help="""
                            - 'event2018' : Event2018 dataset;
                            - 'event2018_image' : all tweets in Event2018 that contain an image. We include tweets that
                        quote an image or reply to an image;
                            - any other value should be the path to your own dataset in tsv format;
                        """)
    parser.add_argument('--model',
                        nargs='+',
                        required=True,
                        choices=text_embeddings + image_embeddings,
                        help="""
                        Choose one text embedding AND/OR one image embedding
                        """
                        )
    parser.add_argument("--save", dest="save", default=False,
                        action="store_true",
                        help="""
                        Save the matrix on disk (.npy format for dense matrix, .npz for sparse matrix)
                        """)
    parser.add_argument("--svd", dest="svd", default=False,
                        action="store_true",
                        help="""
                        Only for TfIdf embedding: create a dense matrix of shape (n_documents, 100)
                        using Singular Value Decomposition
                        """)
    parser.add_argument('--binary', dest="binary", default=True,
                        action="store_false",
                        help="""
                        Only for TfIdf embedding: if True, all non-zero term counts are set to 1.
                        This does not mean outputs will have only 0/1 values, only that the tf term
                        in tf-idf is binary.
                        """)

    parser.add_argument("--hashtag_split", dest="hashtag_split", default=False,
                        action="store_true",
                        help="""
                        Split hashtags into words on capital letters (#FollowFriday --> Follow Friday)
                        """)
    parser.add_argument("--tfidf_weights", dest="tfidf_weights", default=False,
                        action="store_true",
                        help="""
                        Only for w2v embedding: each word vector of each document is weighted with
                        its tfidf weight
                        """)
    parser.add_argument('--text+',
                        dest="text+", default=False,
                        action="store_true",
                        help="""
                        Only if --dataset argument is set to "event2018" or "event2018_image"
                        Use the text of the tweet + the text of the tweet quoted or replied to
                        """)
    parser.add_argument('--annotation',
                        required=False,
                        default="annotated",
                        choices=["annotated", "examined"],
                        help="""
                        Only if --dataset argument is set to "default" or "has_image"
                            - annotated : (default) all tweets annotated as related to an event;
                            - examined : all tweets annotated as related or unrelated to an event;
                        """
                        # - no : all tweets in the dataset regardless of annotation
                        )
    parser.add_argument('--lang',
                        required=False,
                        default="fr",
                        choices=["fr", "en"]
                        )

    args = vars(parser.parse_args())
    for m in args["model"]:
        args1 = args.copy()
        X = []
        args1["model"] = m
        matrix = build_matrix(**args1)
