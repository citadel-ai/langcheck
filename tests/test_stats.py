import pytest

from langcheck.stats import compute_stats

################################################################################
# Tests
################################################################################


@pytest.mark.parametrize(
    'input_text,num_sentences,num_words,num_syllables',
    [
        (
            'My Friend. Welcome to the Carpathians. I am anxiously expecting you.\n'  # NOQA: E501
            'Sleep well to-night. At three to-morrow the diligence will start for Bukovina;\n'  # NOQA: E501
            'a place on it is kept for you.',
            5,
            31,
            46),
        (
            'How slowly the time passes here, encompassed as I am by frost and snow!\n'  # NOQA: E501
            'Yet a second step is taken towards my enterprise.',
            2,
            23,
            32),
    ])
def test_compute_stats(input_text, num_sentences, num_words, num_syllables):
    stats = compute_stats(input_text)
    assert (stats.num_sentences == num_sentences)
    assert (stats.num_words == num_words)
    assert (stats.num_syllables == num_syllables)
