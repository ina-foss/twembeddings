from twython import Twython, TwythonError, TwythonAuthError, TwythonRateLimitError
import time
import logging
import json
import pandas as pd
import argparse
import csv

logging.basicConfig(format='%(asctime)s - %(levelname)s : %(message)s', level=logging.INFO)
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
for argument in ['--path', '--app_key', '--app_secret', '--oauth_token', '--oauth_token_secret']:
    parser.add_argument(argument, required=True)
parser.add_argument('--dataset', required=True, help="event2012, event2018")
args = vars(parser.parse_args())


def format_tweet(row):
    tweet_dict = {"text": row["full_text"], "id": row["id_str"], "created_at": row["created_at"]}
    if "extended_entities" in row and "media" in row["extended_entities"]:
        tweet_dict["url_image"] = row["extended_entities"]["media"][0]["media_url"]
    return tweet_dict


def rehydrate_tweets(twitter_obj, tweet_id_list, jsondump=False):
    logging.info("Starting calls to Twitter API. This may take some time.")
    tweets = []
    id_count = 0
    while True:
        try:
            batch = twitter_obj.lookup_status(
                id=tweet_id_list[id_count:id_count+100],
                include_entities=True,
                tweet_mode="extended")
        except TwythonRateLimitError:
            reset_time = float(twitter_obj.get_lastfunction_header("x-rate-limit-reset"))
            delta = round(int(reset_time) - time.time(), 0)
            logging.warning("Twitter rate limit reached, sleeping {} seconds".format(delta + 1))
            time.sleep(delta + 1)
            continue
        if len(batch) == 0:
            break
        if jsondump:
            with open("data/event_2018.json", "w") as f:
                for tweet in batch:
                    json_str = json.dumps(tweet) + "\n"
                    f.write(json_str)
        tweets += batch
        id_count += 100
        logging.info(" ... {} / {} tweets retrieved so far".format(id_count, len(tweet_id_list)))

    # Check to see if we didn't get all of the requested tweets back
    if len(tweet_id_list) != len(tweets):
        logging.info("{}% of ids collected. Some tweets/accounts may have been deleted.".format(
            round(100*len(tweets)/len(tweet_id_list), 0))
        )

    return tweets


def main():
    path = args.pop("path")
    dataset = args.pop("dataset")
    twitter_obj = Twython(**args)
    labeled_data = pd.read_csv(path, sep="\t", header=None, names=["label", "id"],
                               dtype={"id": str}
                               ).drop_duplicates()
    ids = labeled_data.id.tolist()
    tweets = rehydrate_tweets(twitter_obj, ids, jsondump=True)
    complete_data = pd.DataFrame([format_tweet(row) for row in tweets])
    complete_data = labeled_data.merge(complete_data, on="id", how="left")
    complete_data = complete_data[complete_data.text.notna()]
    complete_data["label"] = complete_data["label"]
    complete_data["text"] = complete_data["text"].str.replace("\t", " ").str.replace("\n", " ").str.replace("\r", " ")
    complete_data.to_csv("data/{}.tsv".format(dataset), sep="\t", index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    main()
