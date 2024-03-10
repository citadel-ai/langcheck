import importlib.util
import sys

from langcheck.augment.en import (change_case, gender, keyboard_typo, ocr_typo,
                                  remove_punctuation, rephrase, synonym)


def lazy_import(name):
    '''Lazily import the language-specific packages in langcheck.augment.

    This prevents `import langcheck` from throwing ModuleNotFoundError if the
    user hasn't installed `langcheck[ja]`, while still allowing the package
    `langcheck.augment.ja` to be visible even if the user didn't explicitly run
    `import langcheck.augment.ja`.

    Copied from: https://docs.python.org/3/library/importlib.html#implementing-lazy-imports  # NOQA: E501
    '''
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None  # For type checking
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


# Use lazy import instead of directly importing language-specific packages
en = lazy_import("langcheck.augment.en")
ja = lazy_import("langcheck.augment.ja")

__all__ = [
    "change_case", "en", "gender", "ja", "keyboard_typo", "ocr_typo", "synonym",
    "remove_punctuation", "rephrase"
]
