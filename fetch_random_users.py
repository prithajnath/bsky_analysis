"""
python fetch_random_users.py -l 100 -b 1000
"""

from bsky import Actor
from network import RetryException
import argparse

import signal


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-l", "--limit", type=int)
    args = parser.parse_args()

    batch_size = args.batch_size or 1000
    limit = args.limit or 100

    actor_api = Actor(batch_size=batch_size, limit=limit)

    signal.signal(signal.SIGINT, actor_api.signal_cleanup)

    try:
        actor_api.get_user_profiles()
    except Exception as e:
        print(f"Caught some exception, flushing buffer {len(actor_api.actors)}")
        actor_api.cleanup()

        raise e
