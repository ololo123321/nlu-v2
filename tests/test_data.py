import pytest
from src.data.base import get_entity_label


@pytest.mark.parametrize("tag, expected", [
    pytest.param("O", "O"),
    pytest.param("I-PER", "PER"),
    pytest.param("B-FOO-BAR", "FOO-BAR"),
    pytest.param("PER", "PER")
])
def test_get_entity_label(tag, expected):
    actual = get_entity_label(tag)
    assert actual == expected
