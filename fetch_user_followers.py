"""
python fetch_user_followers.py -u rmayemsinger.bsky.social
"""

from bsky import ActorFollower
import argparse

import signal


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-u", "--user", required=True)
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-l", "--limit", type=int)
    args = parser.parse_args()

    handle = args.user
    batch_size = args.batch_size or 1000
    limit = args.limit or 100

    actor_follower_api = ActorFollower(
        handle=handle, limit=limit, batch_size=batch_size
    )

    signal.signal(signal.SIGINT, actor_follower_api.signal_cleanup)

    try:
        actor_follower_api.get_followers()
    except Exception as e:
        print(
            f"Caught some exception, flushing buffer {len(actor_follower_api.followers)}"
        )
        actor_follower_api.cleanup()

        raise e
