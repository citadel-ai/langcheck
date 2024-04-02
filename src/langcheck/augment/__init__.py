from langcheck.augment import en
from langcheck.augment.en import (change_case, gender, keyboard_typo, ocr_typo,
                                  remove_punctuation, rephrase, synonym)

__all__ = [
    "change_case", "en", "gender", "keyboard_typo", "ocr_typo", "synonym",
    "remove_punctuation", "rephrase"
]

# Try to import language-specific packages. These packages will be hidden if
# the user didn't pip install the required language.
try:
    from langcheck.augment import ja
except ModuleNotFoundError:
    pass
else:
    __all__.append('ja')
