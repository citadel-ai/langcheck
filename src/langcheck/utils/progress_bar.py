from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tqdm import tqdm


def tqdm_wrapper(
    iterable: Iterable[Any],
    desc: str | None = None,
    total: int | None = None,
    unit: str = "it",
):
    """
    Wrapper for tqdm to make it optional
    """
    if desc is None:
        desc = "Progress"
    if total is None:
        total = len(list(iterable))
    return tqdm(iterable, desc=desc, total=total, unit=unit)
