import logging
import pandas as pd
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import OrdinalEncoder
from collections import Counter
from sklearn import metrics

__all__ = ['general_statistics', 'cluster_event_match', 'mcminn_eval', 'cluster_acc']

def vizualize(vectors, data):
    labels = data.label.unique()
    print(len(labels))
    inter_dist = []
    intra_dist = []
    max_dist = []
    # avg_dist = np.zeros((labels.size, labels.size))
    # max_dist = np.zeros((labels.size, labels.size))

    for i, ilabel in enumerate(labels):
        t0 = time.time()
        for j, jlabel in enumerate(labels):
            if i <= j:
                pairwise_distance = metrics.pairwise_distances(
                    vectors[(data.label == ilabel)],
                    vectors[(data.label == jlabel)],
                    metric="cosine"
                )
                mean_pairwise_distance = pairwise_distance.mean()
                max_pairwise_distance = pairwise_distance.max()
                # avg_dist[i, j] = mean_pairwise_distance
                # max_dist[i, j] = max_pairwise_distance
                max_dist.append(max_pairwise_distance)
                if i < j:
                    inter_dist.append(mean_pairwise_distance)
                elif i == j:
                    # max_pairwise_distance = metrics.pairwise_distances(
                    #     vectors[(data.label == ilabel).values],
                    #     vectors[(data.label == jlabel).values],
                    #     metric=metric
                    # ).max()
                    intra_dist.append(mean_pairwise_distance)
                    logging.info("mean intra_distance event {}: {}".format(i, mean_pairwise_distance))
                    logging.info("max intra_distance event {}: {}".format(i, max_pairwise_distance))
    logging.info("max inter distance: {}".format(np.array(max_dist).max()))
    logging.info("mean inter distance: {}".format(np.array(inter_dist).mean()))
    logging.info("mean intra distance: {}".format(np.array(intra_dist).mean()))


def cluster_acc(data, pred):
    """
    Calculate clustering accuracy.
    (Taken from https://github.com/XifengGuo/IDEC-toy/blob/master/DEC.py)
    # Arguments
        data: pd.DataFrame with shape `(n_samples,n_columns)` with a "label" column containing true labels
        pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    data["pred"] = pd.Series(pred, dtype=data.label.dtype)
    data = data[data.label.notna()].as_matrix(columns=["label", "pred"])
    enc = OrdinalEncoder()
    data = enc.fit_transform(data)
    y_true = data[:,0].astype(np.int64)
    y_pred = data[:,1].astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w) # Optimal label mapping based on the Hungarian algorithm

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def general_statistics(pred):
    s = pd.Series(pred)
    stats = s.groupby(by=s.values).size().describe(percentiles=[])
    return dict(stats)


def cluster_event_match(data, pred):
    data["pred"] = pd.Series(pred, dtype=data.label.dtype)
    data = data[data.label.notna()]
    logging.info("{} labels, {} preds".format(len(data.label.unique()), len(data.pred.unique())))
    t0 = time.time()

    match = data.groupby(["label", "pred"], sort=False).size().reset_index(name="a")
    b, c = [], []
    for idx, row in match.iterrows():
        b_ = ((data["label"] != row["label"]) & (data["pred"] == row["pred"]))
        b.append(b_.sum())
        c_ = ((data["label"] == row["label"]) & (data["pred"] != row["pred"]))
        c.append(c_.sum())
    logging.info("match clusters with events took: {} seconds".format(time.time() - t0))
    match["b"] = pd.Series(b)
    match["c"] = pd.Series(c)
    # recall = nb true positive / (nb true positive + nb false negative)
    match["r"] = match["a"] / (match["a"] + match["c"])
    # precision = nb true positive / (nb true positive + nb false positive)
    match["p"] = match["a"] / (match["a"] + match["b"])
    match["f1"] = 2 * match["r"] * match["p"] / (match["r"] + match["p"])
    match = match.sort_values("f1", ascending=False)
    macro_average_f1 = match.drop_duplicates("label").f1.mean()
    macro_average_precision = match.drop_duplicates("label").p.mean()
    macro_average_recall = match.drop_duplicates("label").r.mean()
    return macro_average_precision, macro_average_recall, macro_average_f1


def mcminn_eval(data, pred, nb_tweets=5, share_tweets=0.8):
    data["pred"] = pd.Series(pred, dtype=data.label.dtype)
    count_label = data.label.value_counts()
    count_pred = data.pred.value_counts()
    data = data[data.label.isin(count_label[count_label >= nb_tweets].index)]
    labels = data.label.unique().tolist()
    all_labels = len(labels)
    data = data[data.pred != -1]
    data = data[data.pred.isin(count_pred[count_pred >= nb_tweets].index)]
    match = data.groupby(["pred", "label"], sort=False).size().reset_index(name="a")
    a, b = 0, 0
    for p in data.pred.unique():
        candidates = match[match.pred == p]
        main = candidates.a.max()
        total = candidates.a.sum()
        if main/total >= share_tweets:
            a += 1
            try:
                labels.remove(candidates[candidates.a == main].label.values[0])
            except ValueError:
                pass
        else:
            b += 1
    precision = a / (a + b)
    recall = (all_labels - len(labels))/all_labels
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1
