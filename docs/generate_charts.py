import csv
from cycler import cycler
import matplotlib.pyplot as plt
from collections import defaultdict
import os

styling = ( cycler('linestyle', ['-', '--', ':', '-.']) +
               cycler('marker', ['o', '+', '^', '.']) +
               cycler('color', ['c', 'm', 'y', 'k'])
)

measures = ["f1", "bcub_f1"]
def plot_chart(ax, results, title, m):
    ax.set_prop_cycle(styling)
    ax.set_ylim([0.2, 0.9])
    ax.grid()
    ax.title.set_text(title)
    for model, points in sorted(results.items()):
        ax.plot(points[0], points[1], label=model)
        ax.legend()
        ax.set(ylabel=f"{m} score", xlabel="threshold")

def append_row(row, results, measure):
    if row["sub_model"]:
        model = row["model"] + " " + row["sub_model"]
    else:
        model = row["model"]
    results[model][0].append(float(row["t"]))
    results[model][1].append(float(row[measure])) 

for m in measures:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))

    en_results, fr_results = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
    with open(os.path.join("..", "results_clustering.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["lang"] == "en":
                append_row(row, en_results, m)
            if row["lang"] == "fr":
                append_row(row, fr_results, m)

    plot_chart(ax1, en_results, "Event2012 (English corpus)", m)
    plot_chart(ax2, fr_results, "Event2018 (Our corpus)", m)
    plt.savefig(f"charts_{m}.jpg", bbox_inches="tight")

