from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class _BaseGenderPronouns:
    subject: str
    dependent_possessive: str
    object: str
    independent_possessive: str
    reflexive: str


_FEMALE_PRONOUNS = _BaseGenderPronouns(subject='she',
                                       dependent_possessive='her',
                                       object='her',
                                       independent_possessive='hers',
                                       reflexive='herself')

_MALE_PRONOUNS = _BaseGenderPronouns(subject='he',
                                     dependent_possessive='his',
                                     object='him',
                                     independent_possessive='his',
                                     reflexive='himself')

_PLURAL_PRONOUNS = _BaseGenderPronouns(
    subject='they',
    dependent_possessive='their',
    object='them',
    independent_possessive='theirs',
    reflexive='themselves',
)

# Neutral pronouns list were taken from some of the most popular neutral
# pronouns from the following sources:
# https://ofm.wa.gov/sites/default/files/public/shr/Diversity/BRGs/Pronouns%20Compilation%20Best%20Practices.pdf
# https://uwm.edu/lgbtrc/support/gender-pronouns/
_NEO_PRONOUNS_LIST = [
    _BaseGenderPronouns(
        subject='zie',
        dependent_possessive='zir',
        object='zim',
        independent_possessive='zis',
        reflexive='zieself',
    ),
    _BaseGenderPronouns(
        subject='zie',
        dependent_possessive='hir',
        object='hir',
        independent_possessive='hirs',
        reflexive='hirself',
    ),
    _BaseGenderPronouns(
        subject='ze',
        dependent_possessive='hir',
        object='hir',
        independent_possessive='hirs',
        reflexive='hirself',
    ),
    _BaseGenderPronouns(
        subject='ze',
        dependent_possessive='zir',
        object='zim',
        independent_possessive='zis',
        reflexive='zieself',
    ),
    _BaseGenderPronouns(
        subject='sie',
        dependent_possessive='hir',
        object='sie',
        independent_possessive='hirs',
        reflexive='hirself',
    ),
    _BaseGenderPronouns(
        subject='ey',
        dependent_possessive='eir',
        object='em',
        independent_possessive='eirs',
        reflexive='eirself',
    ),
    _BaseGenderPronouns(
        subject='ve',
        dependent_possessive='vis',
        object='ver',
        independent_possessive='vers',
        reflexive='verself',
    ),
    _BaseGenderPronouns(
        subject='tey',
        dependent_possessive='ter',
        object='tem',
        independent_possessive='ters',
        reflexive='terself',
    ),
    _BaseGenderPronouns(
        subject='e',
        dependent_possessive='eir',
        object='em',
        independent_possessive='eirs',
        reflexive='ems',
    ),
    _BaseGenderPronouns(
        subject='per',
        dependent_possessive='per',
        object='per',
        independent_possessive='pers',
        reflexive='perself',
    ),
    _BaseGenderPronouns(
        subject='xe',
        dependent_possessive='xyr',
        object='xem',
        independent_possessive='xyrs',
        reflexive='xyrself',
    ),
    _BaseGenderPronouns(
        subject='fae',
        dependent_possessive='faer',
        object='faer',
        independent_possessive='faers',
        reflexive='faerself',
    ),
    _BaseGenderPronouns(
        subject='ae',
        dependent_possessive='aer',
        object='aer',
        independent_possessive='aers',
        reflexive='aerself',
    )
]

_PRONOUNS_DICT = MappingProxyType({
    "plural": [_PLURAL_PRONOUNS],
    "male": [_MALE_PRONOUNS],
    "female": [_FEMALE_PRONOUNS],
    "neutral": _NEO_PRONOUNS_LIST + [_PLURAL_PRONOUNS],
})
