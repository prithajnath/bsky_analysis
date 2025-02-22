from atproto import Client
from atproto_client.exceptions import ModelError, NetworkError
from datetime import datetime
from time import sleep
import os
import pandas as pd
import json
import sys
import signal
import string

from typing import Tuple
from functools import wraps
from random import uniform

USERNAME = os.getenv("BSKY_USERNAME")
APP_PASSWORD = os.getenv("BSKY_APP_PASSWORD")
client = Client()
client.login(USERNAME, APP_PASSWORD)


# This is just a wrapper class for the retry decorator. Whenever we get an instance of this class it means something went wrong
# in the retry process
class RetryException:
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

    __repr__ = __str__


def retry(exceptions: Tuple, max_retries: int = 3):
    """
    Create retry decorators by passing a list/tuple of exceptions. Can also set max number of retries
    Implements exponential backoff with random jitter
    Usage:
        arithmetic_exception_retry = retry(exceptions=(FloatingPointError, OverflowError, ZeroDivisionError), max_retries=2)

        @arithmetic_exception_retry
        def _calculate_center_of_gravity(mass):
            ...
    """

    def _retry(f):
        @wraps(f)
        def _f_with_retries(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    # If we get an exception that we're not sure about, we simply catch it and log the error in the db
                    if type(e) not in exceptions:
                        raise BaseException(
                            f"""
                        Caught an unknown exception while executing {f.__name__}. Refraining from retries
                        {e}
                        """
                        )
                    retries += 1
                    if retries == max_retries:
                        return RetryException(
                            f"""
                        Failed to execute {f.__name__} despite exponential backoff
                        {e}
                        """
                        )
                    else:
                        backoff_interval = 2**retries
                        jitter = uniform(1, 2)
                        total_backoff = backoff_interval + jitter
                        sys.stderr.write(
                            f"""
                            Error: {e}
                            Retrying {f.__name__} #{retries}. Sleeping for {total_backoff}s"""
                        )
                        sys.stderr.flush()
                        sleep(total_backoff)

        return _f_with_retries

    return _retry


network_exception_retry = retry(
    exceptions=(ConnectionError, ModelError, NetworkError),
    max_retries=12,
)


class BlueskyFetch:
    def __init__(self):
        self.username = os.getenv("BSKY_USERNAME")
        self.password = os.getenv("BSKY_APP_PASSWORD")
        self._client = Client()
        self._client.login(self.username, self.password)

    @property
    def api(self):
        return self._client.app.bsky


class Actor(BlueskyFetch):

    def __init__(self, limit=100, batch_size=100):
        self.limit = limit
        self.batch_size = batch_size
        self.actors = []
        super().__init__()

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
        df = pd.DataFrame(self.actors)
        df.to_csv(f"users_{datetime.now()}.csv", index=False)
        self.actors = []

    # To be passed to signal handler only
    def flush(self, a, b):
        self.flush_actors()
        sys.exit()

    @network_exception_retry
    def get_user_profiles(self, letter=None):

        alphabet = string.ascii_lowercase
        for letter in alphabet:
            print(f"Fetching users with letter {letter} with limit {self.limit}")
            params = {"limit": self.limit, "q": letter}
            cursor = None

            batch_flushed = False
            while not batch_flushed:
                if cursor:
                    params["cursor"] = cursor
                response = self.api.actor.search_actors(params=params)

                if response:
                    if response.cursor:
                        cursor = response.cursor
                    else:
                        break

                for actor in response.actors:
                    self.add_actor(actor, letter)

                    if len(self.actors) >= self.batch_size:
                        print(f"Flushing batch: {len(self.actors)}")
                        self.flush_actors()
                        batch_flushed = True
                        break
