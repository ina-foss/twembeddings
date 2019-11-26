from build_features_matrix import build_matrix, load_dataset
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, euclidean_distances
import pandas as pd
import argparse
import logging
import yaml
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        params["model"] = model
        logging.info("Classification with {} model".format(model))
        if model in options:
            # overwrite standard parameters if specified in options.yaml file
            for opt in options[model]:
                if opt != "threshold":
                    params[opt] = options[model][opt]
                    logging.info("Param '{}' : {}".format(opt, options[model][opt]))
        test_params(**params, seeds=[42, 11, 1008, 2993, 559])


def kernel(X, Y):
    return 1 - abs(euclidean_distances(X, Y))


def classif(X_train, X_test, y_train, y_test, seed):
    clf = SVC(kernel=kernel, C=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    return {"p": precision, "r": recall, "f1": f1, "seed": seed}


def test_params(**params):
    data = load_dataset(dataset=params["dataset"], annotation=params["annotation"], text=params["text+"])
    X = build_matrix(**params)
    y = data.label.astype(int).values
    splits = []
    for seed in params.pop("seeds"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=seed)
        splits.append((X_train, X_test, y_train, y_test, seed))

    p = Pool(processes=4)
    results = p.starmap(classif, splits)
    params["kernel"] = "triangular"
    for result in results:
        current_results = params.copy()
        current_results.update(result)
        current_results = pd.DataFrame(current_results, index=[0])
        if params["save_results"]:
            try:
                results = pd.read_csv("results_classif.csv")
            except FileNotFoundError:
                results = pd.DataFrame()
            current_results = results.append(current_results, ignore_index=True)
            current_results.to_csv("results_classif.csv")
    results = pd.DataFrame(results, index=range(len(results)))
    mean = results[["f1", "p", "r"]].mean().round(2)
    std = results[["f1", "p", "r"]].std().round(2)
    logging.info("F1: {}±{}   precision: {}±{}   recall: {}±{}".format(
        mean["f1"], std["f1"], mean["p"], std["p"], mean["r"], std["r"]
    ))


if __name__ == '__main__':
    args = vars(parser.parse_args())
    main(args)


