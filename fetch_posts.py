"""
python fetch_posts.py -b 1000 -i 10:1000
"""

import argparse
import signal

import pandas as pd

from bsky import Actor_Posts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-l", "--limit", type=int)
    parser.add_argument("-i", "--iloc")
    args = parser.parse_args()

    batch_size = args.batch_size or 1000
    limit = args.limit or 100
    iloc = args.iloc

    users_df = pd.read_csv("sample_users.csv")
    if iloc:
        s, e = [int(i) for i in iloc.split(":")]
        print(f"Slicing users df {s}:{e}")
        users_df = users_df.iloc[s:e]

    print(users_df)
    dids = users_df["did"]

    # test DIDs
    # mcuban.bsky.social did:plc:y5xyloyy7s4a2bwfeimj7r3b
    # bodegacats.bsky.social: did:plc:qhfo22pezo44fa3243z2h4ny
    # dummy_did = "did:plc:y5xyloyy7s4a2bwfeimj7r3b"
    # dids = [dummy_did]

    for did in dids:
        feed_api = Actor_Posts(did=did, limit=10, batch_size=100)

        signal.signal(signal.SIGINT, feed_api.signal_cleanup)
        try:
            feed_api.get_user_posts()
        except Exception as e:
            print(f"Caught some exception, flushing buffer {len(feed_api.posts)}")
            feed_api.cleanup()

            raise e
