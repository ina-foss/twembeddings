from twembeddings import build_matrix
from twembeddings import ClusteringAlgo, ClusteringAlgoSparse
from twembeddings import general_statistics, cluster_event_match, mcminn_eval, bcubd_eval

from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import pandas as pd
import logging
import yaml
import argparse
import csv
# from sklearn.cluster import DBSCAN

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
text_embeddings = ['tfidf_dataset', 'tfidf_all_tweets', 'w2v_gnews_en', "elmo", "bert", "sbert", "use"]
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

parser.add_argument('--annotation',
                    required=False,
                    choices=["examined", "annotated", "no"])

parser.add_argument('--threshold',
                    nargs='+',
                    required=False
                    )

parser.add_argument('--batch_size',
                    required=False,
                    type=int
                    )

parser.add_argument('--remove_mentions',
                    action='store_true'
                    )

parser.add_argument('--window',
                    required=False,
                    default=24,
                    type=int
                    )
parser.add_argument('--sub-model',
                    required=False,
                    type=str
                    )

def main(args):
    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)
    for model in args["model"]:
        # load standard parameters
        params = options["standard"]
        logging.info("Clustering with {} model".format(model))
        if model in options:
            # change standard parameters for this specific model
            for opt in options[model]:
                params[opt] = options[model][opt]
        for arg in args:
            if args[arg] is not None:
                # params from command line overwrite options.yaml file
                params[arg] = args[arg]

        params["model"] = model
        test_params(**params)


def test_params(**params):
    X, data = build_matrix(**params)
    params["window"] = int(data.groupby("date").size().mean()*params["window"]/24// params["batch_size"] * params["batch_size"])
    logging.info("window size: {}".format(params["window"]))
    params["distance"] = "cosine"
    # params["algo"] = "DBSCAN"
    # params["min_samples"] = 5
    thresholds = params.pop("threshold")
    for t in thresholds:
        logging.info("threshold: {}".format(t))
        # clustering = DBSCAN(eps=t, metric=params["distance"], min_samples=params["min_samples"]).fit(X)
        if params["model"].startswith("tfidf") and params["distance"] == "cosine":
            clustering = ClusteringAlgoSparse(threshold=float(t), window_size=params["window"],
                                              batch_size=params["batch_size"], intel_mkl=False)
        else:
            clustering = ClusteringAlgo(threshold=float(t), window_size=params["window"],
                                        batch_size=params["batch_size"],
                                        distance=params["distance"])
        clustering.add_vectors(X)
        y_pred = clustering.incremental_clustering()
        # y_pred = clustering.labels_
        stats = general_statistics(y_pred)
        p, r, f1 = cluster_event_match(data, y_pred)
        ami = adjusted_mutual_info_score(data.label, y_pred)
        ari = adjusted_rand_score(data.label, y_pred)
        data["pred"] = data["pred"].astype(int)
        data["id"] = data["id"].astype(int)
        candidate_columns = ["date", "time", "label", "pred", "user_id_str", "id"]
        result_columns = []
        for rc in candidate_columns:
            if rc in data.columns:
                result_columns.append(rc)
        data[result_columns].to_csv(params["dataset"].replace(".", "_results."),
                                    index=False,
                                    sep="\t",
                                    quoting=csv.QUOTE_NONE
                                    )
        try:
            mcp, mcr, mcf1 = mcminn_eval(data, y_pred)
        except ZeroDivisionError as error:
            logging.error(error)
            continue
        try:
            bcp, bcr, bcf1 = bcubd_eval(data, y_pred)
        except ZeroDivisionError as error:
            logging.error(error)
            continue
        stats.update({"t": t, "p": p, "r": r, "f1": f1, "mcp": mcp, "mcr": mcr, "mcf1": mcf1, "ami": ami, "ari": ari, "bcub_p" : bcp, "bcub_r" : bcr, "bcub_f1" : bcf1})
        stats.update(params)
        stats = pd.DataFrame(stats, index=[0])
        logging.info(stats[["t", "model", "tfidf_weights", "p", "r", "f1", "ami", "ari", "bcub_p", "bcub_r", "bcub_f1"]].iloc[0])
        if params["save_results"]:
            try:
                results = pd.read_csv("results_clustering.csv")
            except FileNotFoundError:
                results = pd.DataFrame()
            stats = pd.concat([results, stats], ignore_index=True)
            stats.to_csv("results_clustering.csv", index=False)
            logging.info("Saved results to results_clustering.csv")


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)