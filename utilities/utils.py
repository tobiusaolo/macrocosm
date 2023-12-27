import functools
import multiprocessing


def run_in_subprocess(func: functools.partial, ttl: int):
    """_summary_

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        _type_: _description_
    """

    def wrapped_func(func: functools.partial, queue: multiprocessing.Queue):
        try:
            result = func()
            queue.put(result)
        except Exception as e:
            # Catch exceptions here to add them to the queue.
            queue.put(e)
            pass

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapped_func, args=[func, queue])

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
