from atproto import Client
from atproto_client.exceptions import ModelError
from datetime import datetime
from time import sleep
import os
import pandas as pd
import json
import sys
import signal


USERNAME = os.getenv("BSKY_USERNAME")
APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")
client = Client()
client.login(USERNAME, APP_PASSWORD)


LIMIT = 100

users = {"handle": [], "bio": [], "created_at": []}


# https://docs.bsky.app/docs/api/app-bsky-actor-search-actors
def fetch_user_profiles():
    cursor = None
    params = {"limit": 100, "q": "a"}
    while True:
        if cursor:
            params["cursor"] = cursor
            print(f"Batch {cursor}")
            sleep(5)

        try:
            response = client.app.bsky.actor.search_actors(params=params)

        except ModelError:
            print("Skipping cause bad data")
            cursor = str(int(cursor) + LIMIT)
            continue

        if response.cursor:
            cursor = response.cursor
        else:
            break

        data = json.loads(response.model_dump_json())
        for actor in data["actors"]:
            users["handle"].append(actor["handle"])
            users["bio"].append(actor["description"])
            users["created_at"].append(actor["created_at"])


def write_to_file(signum, frame):
    df = pd.DataFrame(users)
    df.to_csv("users.csv", index=False)

    sys.exit()


if __name__ == "__main__":

    signal.signal(signal.SIGINT, write_to_file)
    fetch_user_profiles()
