"""
Usage
python fetch_user_posts_and_reposts.py -hdl arindube.bsky.social
"""

from atproto import Client
from atproto_client.exceptions import ModelError
from datetime import datetime
from time import sleep
import os
import pandas as pd
import json
import sys
import signal
import argparse


USERNAME = os.getenv("BSKY_USERNAME")
APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")
client = Client()
client.login(USERNAME, APP_PASSWORD)


LIMIT = 100

user_posts = {"handle": [], "created_at": [], "uri": []}


# https://docs.bsky.app/docs/api/app-bsky-feed-get-actor-likes
def fetch_user_posts_and_reposts(handle: str):
    cursor = None
    params = {"limit": LIMIT, "actor": handle}
    while True:
        if cursor:
            params["cursor"] = cursor
            # cursor = str(int(cursor) + LIMIT)
            print(f"Batch {cursor}")
            sleep(5)

        try:
            response = client.app.bsky.feed.get_author_feed(params=params)

        except ModelError:
            print("Skipping cause bad data")
            continue

        # data = json.loads(response.model_dump_json())
        for entry in response.feed:
            post = entry.post
            uri = post.uri
            indexed_at = post.indexed_at
            user_posts["handle"].append(handle)
            user_posts["uri"].append(uri)
            user_posts["created_at"].append(indexed_at)

        if response.cursor:
            cursor = response.cursor
        else:
            break

    write_to_file(1, 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-hdl", "--handle", required=True)
    args = parser.parse_args()

    handle = args.handle

    def write_to_file(signum, frame):
        df = pd.DataFrame(user_posts)
        df.to_csv(f"user_posts_{handle}.csv", index=False)

        sys.exit()

    signal.signal(signal.SIGINT, write_to_file)
    fetch_user_posts_and_reposts(handle=handle)
