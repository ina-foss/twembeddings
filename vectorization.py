import sys
import csv
import casanova
import math
import argparse
from multiprocessing import Pool
from fog.tokenizers import WordTokenizer
from twembeddings.stop_words import STOP_WORDS_EN, STOP_WORDS_FR
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from collections import Counter

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('column', help='Name of the column containing the tweet\'s text.')
parser.add_argument('tweets', help='CSV file containing the tweets to tokenize.')
parser.add_argument(
    '--tsv',
    action='store_true',
    help='Whether the input file is using tabs as separator.'
)

cli_args = parser.parse_args()

def prepare_stoplist():
    stoplist = set()

    for word in STOP_WORDS_EN + STOP_WORDS_FR:
        stoplist.add(word)
        stoplist.add(word + "'")
        stoplist.add(word + "’")
        stoplist.add("'" + word)
        stoplist.add("’" + word)

    return stoplist

tokenizer = WordTokenizer(
    keep=['word'],
    lower=True,
    unidecode=True,
    split_hashtags=True,
    stoplist=prepare_stoplist(),
    reduce_words=True,
    decode_html_entities=True
)

DOCUMENTS = []
DOCUMENT_FREQUENCIES = Counter()

def tokenize(text):
    try:
        return set(value for _, value in tokenizer(text))
    except KeyboardInterrupt:
        sys.exit(1)

with open(cli_args.tweets) as f:
    reader = casanova.reader(f, delimiter='\t' if cli_args.tsv else ',', prebuffer_bytes=3_000_000)

    loading_bar = tqdm(unit='tweet', total=reader.total)

    with Pool(8) as pool:
        for tokens in pool.imap(tokenize, reader.cells(cli_args.column)):
            loading_bar.update()

            for token in tokens:
                DOCUMENT_FREQUENCIES[token] += 1

            DOCUMENTS.append(tokens)

loading_bar.close()
print('Size of vocabulary:', len(DOCUMENT_FREQUENCIES))

print('Most frequent tokens:')
for token, count in DOCUMENT_FREQUENCIES.most_common(50):
    print('  -', token, count, count / len(DOCUMENTS))

N = len(DOCUMENTS)
ID = 0
TOKEN_IDS = {}
INVERSE_DOCUMENT_FREQUENCIES = Counter()

for token, df in DOCUMENT_FREQUENCIES.items():
    if df > 10:
        TOKEN_IDS[token] = ID
        ID += 1
        INVERSE_DOCUMENT_FREQUENCIES[token] = 1 + math.log((N + 1) / (df + 1))

print('Size of vocabulary after df trimming:', len(INVERSE_DOCUMENT_FREQUENCIES))

writer = csv.writer(DummyTqdmFile(sys.stdout))
writer.writerow(['dimensions', 'weights'])

for doc in tqdm(DOCUMENTS):
    vector = [
        (TOKEN_IDS[token], INVERSE_DOCUMENT_FREQUENCIES[token])
        for token in doc
        if token in TOKEN_IDS
    ]

    norm = math.sqrt(sum(w * w for _, w in vector))
    vector = [(_id, w / norm) for _id, w in vector]
    vector = sorted(vector, key=lambda t: (t[1], t[0]), reverse=True)

    writer.writerow([
        '|'.join(str(dim) for dim, _ in vector),
        '|'.join(str(weight) for _, weight in vector)
    ])
