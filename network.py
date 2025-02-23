from functools import wraps
from random import uniform
from typing import Tuple

import sys
from time import sleep


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
                        raise e

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
