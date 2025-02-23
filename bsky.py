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
        self.dbfilename = "file.db"
        self.cursor = None

        # Init db with tables
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS users
                    (
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

    def __init__(self, limit=100, batch_size=100):
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
            # df.to_csv(f"users_{datetime.now()}.csv", index=False)

            if df.shape[0] > 0:
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

        # check for previously saved cursor.
        if self.letter:
            idx = alphabet.index(self.letter)
            print(f"Picking up from letter {self.letter}")
            alphabet = alphabet[idx:]

        for letter in alphabet:
            self.letter = letter
            print(f"Fetching users with letter {self.letter} with limit {self.limit}")
            params = {"limit": self.limit, "q": self.letter}

            batch_flushed = False
            while not batch_flushed:
                if self.cursor:
                    params["cursor"] = self.cursor
                response = self.api.actor.search_actors(params=params)

                if response:
                    if response.cursor:
                        self.cursor = response.cursor
                    else:
                        break

                for actor in response.actors:
                    self.add_actor(actor, letter)

                    if len(self.actors) >= self.batch_size:
                        print(f"Flushing batch: {len(self.actors)}")
                        self.flush_actors()
                        batch_flushed = True
                        break

            # reset cursor if done with while loop
            self.cursor = None
        else:
            # if everything goes well
            self.cursor = "FINISHED"
            self.save_progress()
