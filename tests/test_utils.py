import collections
import pytest
from cophi.text import utils


PARAGRAPHS = [["A B C D E F", "G H I J K L"]]
DOCUMENT = PARAGRAPHS[0][0]
TOKENS = DOCUMENT.split(" ")

def test_construct_ngrams():
    ngrams = utils.construct_ngrams(TOKENS)
    assert list(ngrams) == ["A B", "B C", "C D", "D E", "E F"]

def test_find_tokens():
    tokens = utils.find_tokens(DOCUMENT, r"\w")
    assert list(tokens) == ["A", "B", "C", "D", "E", "F"]
    # Stop tokenizing after the first token:
    tokens = utils.find_tokens(DOCUMENT, r"\w", 1)
    assert list(tokens) == ["A"]

def test_lowercase_tokens():
    tokens = utils.lowercase_tokens(TOKENS)
    assert tokens == ["a", "b", "c", "d", "e", "f"]

def test_segment_fuzzy():
    segments = utils.segment_fuzzy(PARAGRAPHS, 1)
    assert list(segments) == [[["A B C D E F"]], [["G H I J K L"]]]

def test_parameter():
    parameter = utils._parameter(TOKENS, "sichel_s")
    assert len(parameter) == 2
    parameter = utils._parameter(TOKENS, "honore_h")
    assert len(parameter) == 3
    parameter = utils._parameter(TOKENS, "entropy")
    assert len(parameter) == 2
    parameter = utils._parameter(TOKENS, "ttr")
    assert len(parameter) == 2
