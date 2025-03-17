"""
python fetch_posts.py -l 100 -b 1000
"""

import duckdb
from bsky import Actor, Actor_Posts
from network import RetryException
import argparse
import pandas as pd

import signal


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-l", "--limit", type=int)
    args = parser.parse_args()

    batch_size = args.batch_size or 1000
    limit = args.limit or 100

    # users_df = pd.read_csv("sample_users.csv")
    # print(users_df.shape)
    # dids = users_df['did']

    # test DIDs
    # mcuban.bsky.social did:plc:y5xyloyy7s4a2bwfeimj7r3b
    # bodegacats.bsky.social: did:plc:qhfo22pezo44fa3243z2h4ny
    dummy_did = "did:plc:y5xyloyy7s4a2bwfeimj7r3b"
    dids = [dummy_did]

    for did in dids:
        feed_api = Actor_Posts(did=did, limit=10, batch_size=100)

        try:
            feed_api.get_user_posts()
        except Exception as e:
            print(f"Caught some exception, flushing buffer {len(feed_api.posts)}")
            feed_api.cleanup()

            raise e
