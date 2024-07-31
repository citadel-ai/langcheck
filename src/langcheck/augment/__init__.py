from langcheck.augment import en
from langcheck.augment.en import (
    change_case,
    gender,
    jailbreak_template,
    keyboard_typo,
    ocr_typo,
    payload_splitting,
    remove_punctuation,
    rephrase,
    synonym,
    to_full_width,
)

__all__ = [
    "en",
    "change_case",
    "gender",
    "jailbreak_template",
    "keyboard_typo",
    "ocr_typo",
    "payload_splitting",
    "remove_punctuation",
    "rephrase",
    "synonym",
    "to_full_width",
]

# Try to import language-specific packages. These packages will be hidden if
# the user didn't pip install the required language.
try:
    from langcheck.augment import ja  # NOQA: F401
except ModuleNotFoundError:
    pass
else:
    __all__.append("ja")
