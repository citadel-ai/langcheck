from langcheck.augment.en._change_case import change_case
from langcheck.augment.en._gender._gender import gender
from langcheck.augment.en._keyboard_typo import keyboard_typo
from langcheck.augment.en._ocr_typo import ocr_typo
from langcheck.augment.en._remove_punctuation import remove_punctuation
from langcheck.augment.en._rephrase import rephrase
from langcheck.augment.en._synonym import synonym

__all__ = [
    'change_case', 'gender', 'keyboard_typo', 'ocr_typo', 'synonym',
    'remove_punctuation', 'rephrase'
]
