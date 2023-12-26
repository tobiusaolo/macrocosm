import functools
import multiprocessing
import bittensor as bt
from typing import Any, Callable


def run_in_subprocess(func: functools.partial, ttl: int):
    """_summary_

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for.

    Returns:
        _type_: _description_
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapped_func, args=[func, queue])
    process.start()
    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        bt.logging.error(f"Failed to {func.func.__name__} after {ttl} seconds")
        return None

    result = queue.get()
    return result
