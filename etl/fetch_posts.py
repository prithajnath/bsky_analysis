"""
python fetch_posts.py -b 1000 -i 10:1000 -m 100_000
"""

import argparse
import signal

import pandas as pd
from create_logger import logger
from bsky import Actor_Posts
from time import sleep
from atproto_client.exceptions import BadRequestError

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-l", "--limit", type=int)
    parser.add_argument("-m", "--max_posts", type=int)
    parser.add_argument("-i", "--iloc")
    parser.add_argument("-lv", "--log-level", type=str, default="INFO")

    args = parser.parse_args()

    batch_size = args.batch_size or 1000
    limit = args.limit or 100
    max_posts = args.max_posts or 1000
    iloc = args.iloc
    log_level = args.log_level

    logger.setLevel(log_level)

    users_df = pd.read_csv("sample_users.csv")
    if iloc:
        s, e = [int(i) for i in iloc.split(":")]
        logger.info(f"Slicing users df {s}:{e}")
        users_df = users_df.iloc[s:e]

    logger.info(users_df)
    dids = users_df["did"]

    # test DIDs
    # mcuban.bsky.social did:plc:y5xyloyy7s4a2bwfeimj7r3b
    # bodegacats.bsky.social: did:plc:qhfo22pezo44fa3243z2h4ny
    # dummy_did = "did:plc:y5xyloyy7s4a2bwfeimj7r3b"
    # dids = [dummy_did]

    for did in dids:
        feed_api = Actor_Posts(did=did, limit=10, batch_size=100, max_posts=max_posts)

        signal.signal(signal.SIGINT, feed_api.signal_cleanup)
        try:
            feed_api.get_user_posts()
        except Exception as e:
            if e.__class__ == BadRequestError:
                logger.error(f"Found a bad did {did}. Skipping!")
                sleep(5)
                continue

            print(f"Caught some exception, flushing buffer {len(feed_api.posts)}")
            feed_api.cleanup()

            raise e
