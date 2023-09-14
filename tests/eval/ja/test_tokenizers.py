from typing import List

import pytest

from langcheck.eval.ja._tokenizers import (JanomeTokenizer, MecabTokeninzer,
                                           _JapaneseTokenizer)


@pytest.mark.parametrize('text,expected_tokens', [
    (['頭が赤い魚を食べる猫', ['頭', 'が', '赤い', '魚', 'を', '食べる', '猫']]),
    ('猫が、マットの上に座った。', ['猫', 'が', 'マット', 'の', '上', 'に', '座っ', 'た']),
])
@pytest.mark.parametrize('tokenizer', [JanomeTokenizer, MecabTokeninzer])
def test_janome_tokenizer(text: str, expected_tokens: List[str],
                          tokenizer: _JapaneseTokenizer) -> None:
    tokenizer = tokenizer()
    assert tokenizer.tokenize(text) == expected_tokens
