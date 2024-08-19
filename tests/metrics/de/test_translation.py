from typing import List

import pytest

from langcheck.metrics.de import Translate


@pytest.mark.parametrize(
    "de_text,en_text",
    [
        ([
            "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein.",  # noqa: E501
            "I have no personal opinions, emotions or consciousness."
        ]),
        ([
            "Mein Freund. Willkommen in den Karpaten.",
            "My friend, welcome to the Carpathians."
        ]),
        ([
            "Tokio ist die Hauptstadt von Japan.",
            "Tokyo is the capital of Japan."
        ]),
    ])
def test_translate_de_en(de_text: str, en_text: str) -> None:
    translation = Translate("Helsinki-NLP/opus-mt-de-en")
    assert translation(de_text) == en_text


@pytest.mark.parametrize("en_text,de_text", [
    ("I have no personal opinions, emotions or consciousness.",
     "Ich habe keine persönlichen Meinungen, Emotionen oder Bewusstsein."),
    ("My Friend. Welcome to the Carpathians. I am anxiously expecting you.",
     "Willkommen bei den Karpaten, ich erwarte Sie."),
    ("Tokyo is the capital of Japan.", "Tokio ist die Hauptstadt Japans."),
])
def test_translate_en_de(en_text: str, de_text: List[str]) -> None:
    translation = Translate("Helsinki-NLP/opus-mt-en-de")
    assert translation(en_text) == de_text
