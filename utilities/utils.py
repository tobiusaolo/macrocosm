import functools
import multiprocessing
from typing import Any, Tuple

from model.data import ModelId


def validate_hf_repo_id(repo_id: str) -> Tuple[str, str]:
    """Verifies a Hugging Face repo id is valid and returns it split into namespace and name.

    Raises:
        ValueError: If the repo id is invalid.
    """

    if not repo_id:
        raise ValueError("Hugging Face repo id cannot be empty.")

    if not 3 < len(repo_id) <= ModelId.MAX_REPO_ID_LENGTH:
        raise ValueError(
            f"Hugging Face repo id must be between 3 and {ModelId.MAX_REPO_ID_LENGTH} characters."
        )

    parts = repo_id.split("/")
    if len(parts) != 2:
        raise ValueError(
            "Hugging Face repo id must be in the format <org or user name>/<repo_name>."
        )

    return parts[0], parts[1]


def run_in_subprocess(func: functools.partial, ttl: int) -> Any:
    """Runs the provided function on a subprocess with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run. MUST NOT BE A NESTED FUNCTION
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        try:
            result = func()
            queue.put(result)
        except Exception as e:
            # Catch exceptions here to add them to the queue.
            queue.put(e)

    # Use "fork" (the default on all POSIX except macOS), because pickling doesn't seem
    # to work on "spawn".
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    process = ctx.Process(target=wrapped_func, args=[func, queue])

    process.start()
    process.join(timeout=ttl)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Failed to {func.func.__name__} after {ttl} seconds")

    result = queue.get()

    # If we put an exception on the queue then raise instead of returning.
    if isinstance(result, Exception):
        raise result

    return result
