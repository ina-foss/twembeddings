from twembeddings.embeddings import TfIdf
import argparse
import logging
import os
import sys
import csv

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--lang',
                    required=False,
                    default="fr",
                    choices=["en", "fr"])


if __name__ == '__main__':
    args = parser.parse_args()
    tfidf = TfIdf(lang=args.lang)
    tfidf.load_history(lang=args.lang)
    writer = csv.writer(sys.stdout)
    writer.writerow(["token", "df"])
    writer.writerow(["$", tfidf.n_samples])
    for token, df in zip(tfidf.features_names, tfidf.df):
        writer.writerow([token, int(df)])





