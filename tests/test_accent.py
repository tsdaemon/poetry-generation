import pytest

from poetry.accent import transform_into_rythm_map


@pytest.mark.parametrize("word,index,expected", [
    ("аболіціоністу", 9, "_____'_"),
    ("багатоденного", 7, "___'__")
])
def test_rythm_map_transformation(word, index, expected):
    assert transform_into_rythm_map(word, index) == expected