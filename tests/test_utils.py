import collections

import pytest

import cophi

PARAGRAPHS = [["A B C D E F", "G H I J K L"]]
DOCUMENT = PARAGRAPHS[0][0]
TOKENS = DOCUMENT.split(" ")

def test_construct_ngrams():
    ngrams = cophi.utils.construct_ngrams(TOKENS)
    assert list(ngrams) == ["A B", "B C", "C D", "D E", "E F"]

def test_find_tokens():
    tokens = cophi.utils.find_tokens(DOCUMENT, r"\w")
    assert list(tokens) == ["A", "B", "C", "D", "E", "F"]
    # Stop tokenizing after the first token:
    tokens = cophi.utils.find_tokens(DOCUMENT, r"\w", 1)
    assert list(tokens) == ["A"]

def test_lowercase_tokens():
    tokens = cophi.utils.lowercase_tokens(TOKENS)
    assert tokens == ["a", "b", "c", "d", "e", "f"]

def test_segment_fuzzy():
    segments = cophi.utils.segment_fuzzy(PARAGRAPHS, 1)
    assert list(segments) == [[["A B C D E F"]], [["G H I J K L"]]]

"""
def test_parameter():
    parameter = cophi.utils._parameter(TOKENS, "sichel_s")
    assert parameter == {"num_types": 6, "freq_spectrum": collections.Counter({1: 6})}
    parameter = cophi.utils._parameter(TOKENS, "honore_h")
    assert parameter == {'num_types': 6, 'num_tokens': 6, 'freq_spectrum': Counter({1: 6})}
    parameter = cophi.utils._parameter(TOKENS, "entropy")
    assert parameter == 
    parameter = cophi.utils._parameter(TOKENS, "ttr")
    assert parameter == 
"""