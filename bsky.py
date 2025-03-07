from atproto import Client
from atproto_client.exceptions import ModelError, NetworkError, InvokeTimeoutError
from datetime import datetime
from time import sleep
import os
import pandas as pd
import json
import sys
import signal
import string
import duckdb

from network import retry

USERNAME = os.getenv("BSKY_USERNAME")
APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")
DB_FILENAME = "file.db"

client = Client()
client.login(USERNAME, APP_PASSWORD)

network_exception_retry = retry(
    exceptions=(ConnectionError, ModelError, NetworkError, InvokeTimeoutError),
    max_retries=12,
)


class BlueskyFetch:
    def __init__(self):
        self.username = os.getenv("BSKY_USERNAME")
        self.password = os.getenv("BSKY_APP_PASSWORD")
        self._client = Client()
        self._client.login(self.username, self.password)
        self.dbfilename = DB_FILENAME
        self.cursor = None

        # Init db with tables
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS users
                    (
                        did varchar(100),
                        handle varchar(100),
                        bio text,
                        created_at timestamptz,
                        letter varchar(1)                   
                    );
                """
            )

    @property
    def api(self):
        return self._client.app.bsky

    def save_progress(self):
        pass

    def cleanup(self):
        pass


class Actor(BlueskyFetch):

    def __init__(self, limit=1000, batch_size=1000):
        self.limit = limit
        self.batch_size = batch_size
        self.actors = []
        self.letter = None
        super().__init__()

        # check for existing cursor
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS users_progress
                    (
                        fetched_at timestamptz,
                        cursor varchar(200),
                        letter varchar(1)                  
                    );
                """
            )

            result = conn.execute(
                """
                with cte1 as (
                    select
                         *,
                         row_number() over (order by fetched_at desc) as r
                    from 
                         users_progress         
                ) select cursor, letter from cte1 where r = 1;
            """
            ).fetchone()

            if result:
                latest_cursor, latest_letter = result
                self.cursor = latest_cursor
                self.letter = latest_letter

    def add_actor(self, actor, letter):

        self.actors.append(
            {
                "did": actor.did,
                "handle": actor.handle,
                "bio": actor.description,
                "created_at": actor.created_at,
                "letter": letter,
            }
        )

    def flush_actors(self):
        # backend

        with duckdb.connect(self.dbfilename) as conn:
            df = pd.DataFrame(self.actors)

            if df.shape[0] > 0:
                print(f"Writing {df.shape[0]} new records to the database.")
                conn.execute("INSERT INTO users SELECT * FROM df")
        self.actors = []

    def save_progress(self):
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                "INSERT INTO users_progress(fetched_at, cursor, letter) VALUES (?, ?, ?)",
                (datetime.now(), self.cursor, self.letter),
            )

    def cleanup(self):
        self.save_progress()
        self.flush_actors()

    def signal_cleanup(self, a, b):
        self.cleanup()
        sys.exit()

    @network_exception_retry
    def get_user_profiles(self, letter=None):
        if self.cursor:
            if self.cursor == "FINISHED":
                return

        alphabet = string.ascii_lowercase

        # Resume from last fetched letter if applicable
        if self.letter:
            idx = alphabet.index(self.letter)
            print(f"🔄 Resuming from letter {self.letter}")
            alphabet = alphabet[idx:]

        for letter in alphabet:
            self.letter = letter
            self.actors = []  # Reset list for each letter
            self.cursor = None
            print(f"Fetching users with letter {letter} with limit {self.limit}")

            # Keep fetching until we reach 10,000 users
            while (
                len(self.actors) < self.batch_size
            ):  # Keep fetching until we reach 10,000 users
                # How many more do we need?
                remaining_needed = self.batch_size - len(self.actors)

                #  adjust the limit so we never exceed 10,000
                adjusted_limit = min(self.limit, remaining_needed)

                params = {"limit": adjusted_limit, "q": letter}
                if self.cursor:
                    params["cursor"] = self.cursor

                print(f"Fetching next batch for {letter} (Limit: {adjusted_limit})...")
                response = self.api.actor.search_actors(params=params)

                if not response or not response.actors:
                    print(f"No more users found for {letter}")
                    # Stop fetching if there are no more users
                    break

                # Store the next cursor for pagination
                self.cursor = response.cursor if response.cursor else None

                # Add only the exact number of users needed to complete 10,000
                needed_users = remaining_needed
                users_to_add = response.actors[:needed_users]  # Slice to prevent excess

                for actor in users_to_add:
                    self.add_actor(actor, letter)

                print(
                    f"Collected {len(self.actors)} users for {letter} (Added: {len(users_to_add)})"
                )

                # If we reach the batch size, flush and continue fetching
                if len(self.actors) >= self.batch_size:
                    print(f"Flushing batch for letter {letter}: {len(self.actors)}")
                    self.flush_actors()
                    # Clear batch after writing
                    self.actors = []

                    # Stop fetching since we hit 10,000
                    break

            # Final flush for remaining users of the letter
            if self.actors:
                print(f"Flushing final batch for letter {letter}: {len(self.actors)}")
                self.flush_actors()
                self.actors = []  # Clear batch after writing

        self.cursor = "FINISHED"
        self.save_progress()


if __name__ == "__main__":
    print(f"USERNAME: {USERNAME}")
    print(
        f"APP_PASSWORD: {'*' * len(APP_PASSWORD) if APP_PASSWORD else 'None'}"
    )  # Mask the password for security

    actor_api = Actor(limit=100, batch_size=10000)
    actor_api.get_user_profiles()
