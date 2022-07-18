"""This module contains some useful functions to both user and internal code."""

import time
from contextlib import contextmanager


@contextmanager
def timer(verbose: bool, operation: str, unit: str):
    """Use this function with 'with' to time any block of code."""

    try:
        if verbose:
            print(f"{unit}: starting {operation}.")
        start_time = time.time()
        yield start_time
    finally:
        if verbose:
            print(f"{unit}: {operation} took {round(time.time() - start_time, 2)} seconds.")
