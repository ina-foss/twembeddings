from build_features_matrix import build_matrix, load_dataset, load_matrix
from clustering_algo import ClusteringAlgo
import numpy as np
from eval import general_statistics, cluster_event_match, mcminn_eval
import pandas as pd
import logging
from scipy.sparse import issparse, hstack, csr_matrix
import yaml
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ['tfidf_dataset', 'tfidf_all_tweets', 'w2v_gnews_en', "elmo", "bert", "sbert_nli_sts", "use"]
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model',
                    nargs='+',
                    required=True,
                    choices=text_embeddings,
                    help="""
                    One or several text embeddings
                    """
                    )
parser.add_argument('--dataset',
                    required=True,
                    help="""
                    Path to the dataset
                    """
                    )

parser.add_argument('--lang',
                    required=True,
                    choices=["en", "fr"])


def main(args):
    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)
    for model in args["model"]:
        params = options["standard"]
        logging.info("Clustering with {} model".format(model))
        if model in options:
            # overwrite standard parameters if specified in options.yaml file
            for opt in options[model]:
                params[opt] = options[model][opt]
                logging.info("Param '{}' : {}".format(opt, options[model][opt]))
        for arg in args:
            params[arg] = args[arg]
        params["model"] = model
        test_params(**params)


def test_params(**params):
    data = load_dataset(dataset=params["dataset"], annotation=params["annotation"], text=params["text+"])
    X = load_matrix(**params)
    if X is None:
        X = build_matrix(**params)
    params["batch_size"] = 8
    params["window"] = int(data.groupby("date").size().mean()//params["batch_size"]*params["batch_size"])
    params["distance"] = "cosine"
    thresholds = params.pop("threshold")
    for t in thresholds:
        logging.info("threshold: {}".format(t))
        clustering = ClusteringAlgo(threshold=t, window_size=params["window"], batch_size=params["batch_size"],
                                    distance=params["distance"])
        clustering.add_vectors(X)
        y_pred = clustering.incremental_clustering()
        stats = general_statistics(y_pred)
        p, r, f1 = cluster_event_match(data, y_pred)
        try:
            mcp, mcr, mcf1 = mcminn_eval(data, y_pred)
        except ZeroDivisionError as error:
            logging.error(error)
            continue
        stats.update({"t": t, "p": p, "r": r, "f1": f1, "mcp": mcp, "mcr": mcr, "mcf1": mcf1})
        stats.update(params)
        stats = pd.DataFrame(stats, index=[0])
        logging.info(stats[["t", "model", "tfidf_weights", "p", "r", "f1"]].iloc[0])
        if params["save_results"]:
            try:
                results = pd.read_csv("results_clustering.csv")
            except FileNotFoundError:
                results = pd.DataFrame()
            stats = results.append(stats, ignore_index=True)
            stats.to_csv("results_clustering.csv")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)