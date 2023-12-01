from typing import Any, Iterable, Optional

from tqdm import tqdm


def tqdm_wrapper(iterable: Iterable[Any],
                 desc: Optional[str] = None,
                 total: Optional[int] = None,
                 unit: str = "it"):
    """
    Wrapper for tqdm to make it optional
    """
    if desc is None:
        desc = "Progress"
    if total is None:
        total = len(list(iterable))
    return tqdm(iterable, desc=desc, total=total, unit=unit)
