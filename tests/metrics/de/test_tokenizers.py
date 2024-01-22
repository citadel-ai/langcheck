from typing import List

import pytest

from langcheck.metrics.de import DeTokenizer


@pytest.mark.parametrize('text,expected_tokens', [
    ([
        'Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.',
        [
            'Ich', 'habe', 'keine', 'persönlichen', 'Meinungen', ',',
            'Emotionen', 'oder', 'Bewusstsein', '.'
        ]
    ]),
    ('Mein Freund. Willkommen in den Karpaten. Ich erwarte dich sehnsüchtig.\n',
     [
         'Mein', 'Freund', '.', 'Willkommen', 'in', 'den', 'Karpaten', '.',
         'Ich', 'erwarte', 'dich', 'sehnsüchtig', '.'
     ]),
])
def test_de_tokenizer(text: str, expected_tokens: List[str]) -> None:
    tokenizer = DeTokenizer()  # type: ignore[reportGeneralTypeIssues]
    assert tokenizer.tokenize(text) == expected_tokens
