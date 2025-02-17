import casanova
import argparse
from vllm import LLM
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dangvantuan/sentence-camembert-large")
parser.add_argument("--prompt", default=None)
parser.add_argument("--norm", action="store_true")
parser.add_argument("--vllm", action="store_true")


args = parser.parse_args()

reader = casanova.reader("../data/tweets_pairs_valid_scores.csv")
textpos1 = reader.headers["1_text"]
textpos2 = reader.headers["2_text"]
moypos = reader.headers["Moyenne"]
texts1, texts2, scores = [], [], []
for row in reader:
    texts1.append(row[textpos1])
    texts2.append(row[textpos2])
    score = 1 - float(row[moypos]) / 5.0
    scores.append(score)

if args.vllm:
    model = LLM(
        model=args.model,
        task="embed",
        enforce_eager=True,
        dtype="float32"
    )

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    embeddings_1 = [v.outputs.embedding for v in model.embed(texts1)]
    embeddings_2 = [v.outputs.embedding for v in model.embed(texts2)]


else:
    model = SentenceTransformer(
        args.model,
    )
    prompt = args.prompt + " \n Query: " if args.prompt else None
    embeddings_1 = model.encode(texts1, prompt=prompt, normalize_embeddings=args.norm)
    embeddings_2 = model.encode(texts2, prompt=prompt, normalize_embeddings=args.norm)

distances = paired_cosine_distances(embeddings_1, embeddings_2)

pearson = pearsonr(distances, scores)
print(pearson)
print("pred: ", [round(d, 1) for d in distances[-10:]])
print("true: ", [round(s, 1) for s in scores[-10:]])

str_model = args.model.replace("/", "_")
with open("../data/tweets_pairs_valid_scores.csv") as input, open(f"results_encoding_{str_model}.csv", "w") as output:
    enricher = casanova.enricher(input, output, add=['cosine'])
    for cosine, row in zip(distances, enricher):
        enricher.writerow(row, [cosine])


with open("pearson_coefficients.csv", "a") as f:
    writer = casanova.writer(f, fieldnames=["model", "prompt", "pearson_score", "p_value", "normalize_embeddings", "vllm"], write_header=False)
    writer.writerow([args.model, args.prompt, pearson.statistic, pearson.pvalue, str(args.norm).lower(), str(args.vllm).lower()])
