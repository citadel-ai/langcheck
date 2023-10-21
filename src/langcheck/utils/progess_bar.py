from tqdm import tqdm


def tqdm_wrapper(iterable, desc=None, total=None, unit="it"):
    """
    Wrapper for tqdm to make it optional
    """
    if desc is None:
        desc = "Progress"
    if total is None:
        total = len(list(iterable))
    return tqdm(iterable, desc=desc, total=total, unit=unit)