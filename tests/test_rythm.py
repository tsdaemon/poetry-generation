import pytest

from poetry.accent import read_vocab
from poetry.rythm import transform_lines_into_rythm


@pytest.fixture()
def verse_sentence():
    return """
І жінка з чорним, як земля, волоссям,
яку я знаю вже стільки років,
живе собі, не переймаючись зовсім,
поміж ранкового світла й вечірніх мороків.
    """


@pytest.fixture()
def accent_vocab():
    return read_vocab('./data/accents.csv')


@pytest.fixture()
def verse_rythm():
    return """
''_'_'_'__'
_'''_''_'_
_'_''__'__'_
_'_'___'_'___'
    """


def test_rythm_map_transformation(verse_sentence, accent_vocab, verse_rythm):
    assert transform_lines_into_rythm(verse_sentence, accent_vocab) == verse_rythm