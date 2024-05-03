import csv
from cycler import cycler
import matplotlib.pyplot as plt
from collections import defaultdict

# Create cycler object. Use any styling from above you please
monochrome = ( cycler('linestyle', ['-', '--', ':', '-.']) +
               cycler('marker', ['o', '+', '^', '.']) +
               cycler('color', ['c', 'm', 'y', 'k'])
)

fig, (ax1, ax2) = plt.subplots(1, 2)
en_results, fr_results = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
with open("results_clustering_fr.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["lang"] == "en":
            en_results[row["model"]][0].append(float(row["t"]))
            en_results[row["model"]][1].append(float(row["f1"]))
        if row["lang"] == "fr":
            fr_results[row["model"]][0].append(float(row["t"]))
            fr_results[row["model"]][1].append(float(row["f1"]))

ax1.set_prop_cycle(monochrome)
ax1.grid()
for model, points in en_results.items():
    ax1.plot(points[0], points[1], label=model)
ax1.legend()

ax2.set_prop_cycle(monochrome)
ax2.grid()
for model, points in fr_results.items():
    ax2.plot(points[0], points[1], label=model)
ax2.legend()

plt.savefig("charts.jpg")

