import pkgutil
from typing import List

import pytest

from langcheck.metrics.zh import HanLPTokenizer
from langcheck.metrics.zh._tokenizers import _ChineseTokenizer


@pytest.mark.parametrize('text,expected_tokens', [
    ('吃葡萄不吐葡萄皮。不吃葡萄到吐葡萄皮。',
     ['吃', '葡萄', '不', '吐', '葡萄', '皮', '不', '吃', '葡萄', '到', '吐', '葡萄', '皮']),
    ('北京是中国的首都', ['北京', '是', '中国', '的', '首都']),
])
@pytest.mark.parametrize(
    'tokenizer',
    [HanLPTokenizer])
def test_janome_tokenizer(text: str, expected_tokens: List[str],
                          tokenizer: _ChineseTokenizer) -> None:
    tokenizer = tokenizer()
    assert tokenizer.tokenize(text) == expected_tokens
