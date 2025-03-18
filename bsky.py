import os
import string
import sys
from datetime import datetime

import atproto_client.models.app.bsky.feed.defs as defs
import duckdb
import pandas as pd
from atproto import Client
from atproto_client.exceptions import InvokeTimeoutError, ModelError, NetworkError
from create_logger import logger
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

                    CREATE TABLE IF NOT EXISTS posts
                    (
                        author_did varchar(100),
                        uri varchar(100),
                        cid varchar(100),
                        created_at timestamptz,
                        body text,
                        reply_uri varchar(100),
                        quote_uri varchar(100)
                    )
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
                logger.debug(f"Writing {df.shape[0]} new records to the database.")
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
            logger.debug(f"🔄 Resuming from letter {self.letter}")
            alphabet = alphabet[idx:]

        for letter in alphabet:
            self.letter = letter
            self.actors = []  # Reset list for each letter
            self.cursor = None
            logger.debug(f"Fetching users with letter {letter} with limit {self.limit}")

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

                logger.debug(
                    f"Fetching next batch for {letter} (Limit: {adjusted_limit})..."
                )
                response = self.api.actor.search_actors(params=params)

                if not response or not response.actors:
                    logger.debug(f"No more users found for {letter}")
                    # Stop fetching if there are no more users
                    break

                # Store the next cursor for pagination
                self.cursor = response.cursor if response.cursor else None

                # Add only the exact number of users needed to complete 10,000
                needed_users = remaining_needed
                users_to_add = response.actors[:needed_users]  # Slice to prevent excess

                for actor in users_to_add:
                    self.add_actor(actor, letter)

                logger.debug(
                    f"Collected {len(self.actors)} users for {letter} (Added: {len(users_to_add)})"
                )

                # If we reach the batch size, flush and continue fetching
                if len(self.actors) >= self.batch_size:
                    logger.debug(
                        f"Flushing batch for letter {letter}: {len(self.actors)}"
                    )
                    self.flush_actors()
                    # Clear batch after writing
                    self.actors = []

                    # Stop fetching since we hit 10,000
                    break

            # Final flush for remaining users of the letter
            if self.actors:
                logger.debug(
                    f"Flushing final batch for letter {letter}: {len(self.actors)}"
                )
                self.flush_actors()
                self.actors = []  # Clear batch after writing

        self.cursor = "FINISHED"
        self.save_progress()


class Actor_Posts(BlueskyFetch):
    MAX_POSTS = 1000

    def __init__(self, did, limit=1000, batch_size=1000, max_posts=None):
        self.limit = limit
        self.batch_size = batch_size
        self.total_fetched = 0
        self.did = did
        self.posts = []
        self.max_posts = max_posts or self.MAX_POSTS
        super().__init__()

        # check for existing cursor
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                """
                    CREATE TABLE IF NOT EXISTS posts_progress
                    (
                        fetched_at timestamptz,
                        cursor varchar(200),
                        did varchar(100)
                    );
                """
            )
            result = conn.execute(
                f"""
                with cte1 as (
                    select
                         *,
                         row_number() over (partition by did order by fetched_at desc) as r
                    from 
                         posts_progress
                ) select cursor from cte1 where r = 1 and did='{self.did}';
            """
            ).fetchone()

            if result:
                latest_cursor = result[0]
                self.cursor = latest_cursor

    def add_post(self, post: defs.PostView):
        new_post = {
            "author_did": self.did,  # author DID
            "uri": post.uri,  # uniform resource identifier
            "cid": post.cid,  # content identifier
            "created_at": post.record.created_at,  # UTC timestamp
            "text": post.record.text,  # text contents of post | TODO: process this and obfuscate it
            # "reply_cid": "None" if post.record.reply is None else post.record.reply.parent.cid,
        }

        try:
            new_post["reply_cid"] = post.record.reply.parent.cid
        except:
            new_post["reply_cid"] = "None"

        try:
            new_post["quote_cid"] = post.embed.record.cid
        except:
            new_post["quote_cid"] = "None"

        self.posts.append(new_post)

        # If we reach the batch size, flush and continue fetching
        if len(self.posts) >= self.batch_size:
            logger.debug(
                f"Flushing batch of {len(self.posts)} posts for user {self.did} ({self.total_fetched} total)"
            )
            self.flush_posts()

    def flush_posts(self):
        # backend
        with duckdb.connect(self.dbfilename) as conn:
            df = pd.DataFrame(self.posts)

            if df.shape[0] > 0:
                logger.debug(f"Writing {df.shape[0]} new records to the database.")
                conn.execute("INSERT INTO posts SELECT * FROM df")
        self.posts = []

    def save_progress(self):
        with duckdb.connect(self.dbfilename) as conn:
            conn.execute(
                "INSERT INTO posts_progress(fetched_at, cursor, did) VALUES (?, ?, ?)",
                (datetime.now(), self.cursor, self.did),
            )

    def cleanup(self):
        self.save_progress()
        self.flush_posts()

    def signal_cleanup(self, a, b):
        self.cleanup()
        sys.exit()

    @network_exception_retry
    def get_user_posts(self):
        if self.cursor:
            if self.cursor == "FINISHED":
                return
        while self.total_fetched < self.max_posts:
            logger.debug(f"posts:{len(self.posts)} | batch_size:{self.batch_size}")
            params = {"limit": self.limit, "actor": self.did}
            logger.debug(f"self cursor:{self.cursor}")
            if self.cursor:
                params["cursor"] = self.cursor

            response = self.api.feed.get_author_feed(params=params)

            if not response or not response.feed:
                logger.debug(f"No more posts found for {self.did}")
                # Stop fetching if there are no more posts

            # Store the next cursor for pagination
            if response.cursor:
                self.cursor = response.cursor
            else:
                logger.debug(f"No more posts found for {self.did}")
                break

            for feed_view_post in response.feed:
                self.total_fetched += 1
                self.add_post(feed_view_post.post)

            logger.debug(
                f"Collected {len(self.posts)} posts for {self.did} (Added: {len(response.feed)})"
            )

        # Final flush for remaining posts of the user
        if self.posts:
            logger.debug(f"Flushing final batch for user {self.did}: {len(self.posts)}")
            self.flush_posts()

        logger.info(f"Collected {self.total_fetched} posts for {self.did}")

        self.cursor = "FINISHED"
        self.save_progress()
