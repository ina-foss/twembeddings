"""
Adapted from https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/sts/training_stsbenchmark_continue_training.py

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset_path = '../data/tweets_pairs_valid_scores.csv'

model_name = 'paraphrase-multilingual-mpnet-base-v2'
train_batch_size = 16
num_epochs = 4
model_save_path = '../data/models/tweets_pairs_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")

data = []
with open(dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        score = float(row['Moyenne']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['1_text'], row['2_text']], label=score)
        data.append(inp_example)

train_samples, dev_samples = train_test_split(data, test_size=0.2)


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


# Development set: Measure correlation between cosine score and gold labels
logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=500,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


# ##############################################################################
# #
# # Load the stored model and evaluate its performance on STS benchmark dataset
# #
# ##############################################################################

# model = SentenceTransformer(model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
# test_evaluator(model, output_path=model_save_path)
