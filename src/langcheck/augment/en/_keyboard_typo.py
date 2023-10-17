from typing import List

import nlpaug.augmenter.char as nac


def keyboard_typo(
    texts: List[str],
    **kwargs,
) -> List[str]:
    aug = nac.KeyboardAug()
    return aug.augment(texts, **kwargs)
