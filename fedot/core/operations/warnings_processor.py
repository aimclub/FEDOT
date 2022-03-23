import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    """ Doesn't display messages in the terminal. According to solution
    http://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
