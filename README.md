# Sentence embeddings for unsupervised event detection in the Twitter stream

This repo aims at letting researchers reproduce our Twitter event detection results on the 
[Event2012 dataset](https://cs.adelaide.edu.au/~wei/sublinks/papers/2.1.2013CIKM.Building%20a%20Large-scale%20Corpus%20for%20Evaluating%20Event.pdf).
Since some tweets may have probably been erased since we collected the dataset, we cannot ensure 100% identical results,
 but we are confident that the comparative performance of the models will remain unchanged.
 

## Summary:
* [Installation](#installation)
* [Download Event2012 dataset](#download-event2012-dataset)
    * [Download tweets' IDs](#download-tweets-ids)
    * [Download tweets' content](#download-tweets-content)
* [Run event detection](#run-event-detection)
* [Available embeddings](#available-embeddings)
    * [tf-idf](#tfidf-dataset)
    * [Word2Vec](#w2v-news)
    * [ELMo](#elmo)
    * [BERT](#bert)
    * [Universal Sentence Encoder](#use)
    * [Sentence-BERT](#sbert)


## Installation
We recommand using Anaconda 3 to create a python 3.6 environment 
(install Anaconda [here](https://docs.anaconda.com/anaconda/install/)):

    conda create -n "twembeddings" python=3.6.9
    source activate twembeddings
    
Then clone the repo and install the complementary requirements:

    cd $HOME
    git clone https://github.com/bmaz/twembeddings.git
    cd twembeddings
    pip install -r requirements.txt
    
## Download Event2012 dataset

### Download tweets' IDs
In compliance with Twitter terms of use, the authors of the dataset do not share the tweets content,
but only the tweets IDs. Accept the 
[dataset agreement](https://docs.google.com/forms/d/e/1FAIpQLSfRQX4R2O_Pv26wuepydKS4xxi6QbLrhaCgJaAXPcKx7dDljQ/viewform)
and download the dataset. Untar the folder, the labeled tweets are in the `relevant_tweets.tsv` file. 

### Download tweets' content
We provide a script to download the content of the tweets from the Twitter API. In order to run the script,
you need to [create a Twitter developper account and a Twitter App](https://developer.twitter.com/en/docs/basics/apps/overview).
Then get the app's [access tokens](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens).
You should now have 4 tokens (the following strings are random examples):
- app_key: mIsU1P0NNjUTf9DjuN6pdqyOF
- app_secret: KAd5dpgRlu0X3yizTfXTD3lZOAkF7x0QAEhAMHpVCufGW4y0t0
- oauth_token: 4087833385208874171-k6UR7OGNFdfBcqPye8ps8uBSSqOYXm
- oauth_token_secret: Z9nZBVFHbIsU5WQCGT7ZdcRpovQm0QEkV4n4dDofpYAEK

Run the script:

    python get_tweets_objects.py \
    --path /yourpath/relevant_tweets.tsv \
    --app_key mIsU1P0NNjUTf9DjuN6pdqyOF \
    --app_secret KAd5dpgRlu0X3yizTfXTD3lZOAkF7x0QAEhAMHpVCufGW4y0t0 \
    --oauth_token 4087833385208874171-k6UR7OGNFdfBcqPye8ps8uBSSqOYXm \
    --oauth_token_secret Z9nZBVFHbIsU5WQCGT7ZdcRpovQm0QEkV4n4dDofpYAEK

The script may take some time to run entirely, since it respects the API's 
[rate limit](https://developer.twitter.com/en/docs/basics/rate-limits).

## Run event detection

## Available embeddings
##### tfidf-dataset: 
Since the same word is rarely used several times in the same tweet, we used
the idf expression rather than the tfidf

![idf(t) = 1+log((n+1)/df(t)+1)](https://latex.codecogs.com/gif.latex?idf(t)=1&plus;log\frac{n&plus;1}{df(t)&plus;1})
##### w2v-news
 [Google model pretrained on google news](https://code.google.com/archive/p/word2vec/) with mean-pooling of word representations as sentence embedding.
<!---
##### w2v-twitter
[Model pretrained on tweets](github.com/loretoparisi/word2vec-twitter) with mean-pooling of word representations as sentence embedding.
--->
##### elmo
Pretrained model on [TensorFlow Hub](https://tfhub.dev/google/elmo/2) with mean-pooling of word representations as sentence embedding. 
##### bert
In case you want to use BERT embeddings, you need to install `bert-as-service`:

    pip install bert-serving-server
    pip install bert-serving-client
    
Then follow the [guidelines](https://github.com/hanxiao/bert-as-service#getting-started) to download a BERT model 
(we used [BERT-Large, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)
in the paper) and start the BERT service:

    bert-serving-start -model_dir=/yourpath/cased_L-24_H-1024_A-16 -max_seq_len=500 -max_batch_size=64 
 
Our program will act as a client to this service.
We use the default parameters of `bert-as-service` : the pooling layer is the second-to-last layer,
and mean-pooling is used for sentence embedding.

##### use
Pretrained model on [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder-large/3)
##### sbert
Pretrained model from [UKPLab](https://github.com/UKPLab/sentence-transformers#pretrained-models). 
We use bert-large-nli-stsb-mean-tokens model.