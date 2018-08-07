"""
cophi_toolbox.utils
~~~~~~~~~~~~~~~~~~~

This module provides low-level helper functions to manage and 
process text data in Python.
"""

from typing import Iterable, Generator, Optional, List, Iterator
import collections

import pandas as pd
import regex as re


def construct_ngrams(tokens: List[str], n: int = 2, sep: str = " ") -> Iterator[str]:
    """
    Parameters:
        tokens: The tokenized document.
        n: Number of tokens per ngram.
        sep: Separator between words within an ngram.
    """
    return (sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)]))

def count_tokens(tokens: Iterable[str]) -> pd.Series:
    """
    Parameters:
        tokens: Tokens to count.
    """
    return pd.Series(collections.Counter(tokens))

def find_tokens(document: str, pattern: str = r"\p{L}+\p{P}?\p{L}+",
                maximum: Optional[int] = None) -> Generator[str, None, None]:
    """
    Parameters:
        document: The content of a text document.
        pattern: A regex pattern for one token.
        maximum: Stop tokenizing after that much tokens.
    """
    count = 0
    for match in re.compile(pattern).finditer(document):
        count += 1
        yield match.group(0)
        if maximum is not None and count >= maximum:
            return

def segment_fuzzy(paragraphs: Iterable[Iterable[str]], segment_size: int = 1000,
                  tolerance: float = 0.05) -> Generator[list, None, None]:
    """Segment a string, respecting paragraphs.

    Parameters:
        paragraphs: Paragraphs of a text document as separated entities.
        segment_size: The target length of each segment in tokens.
        tolerance: How much may the actual segment size differ from the `segment_size`? 
            If ``0 < tolerance < 1``, this is interpreted as a fraction of the `segment_size`, 
            otherwise it is interpreted as an absolute number. If ``tolerance < 0``, paragraphs 
            are never split apart.
    """
    if tolerance > 0 and tolerance < 1:
        tolerance = round(segment_size * tolerance)
    current_segment = []
    carry = None
    current_size = 0
    doc_iter = iter(paragraphs)
    try:
        while True:
            chunk = list(carry if carry else next(doc_iter))
            carry = None
            current_segment.append(chunk)
            current_size += len(chunk)
            if current_size >= segment_size:
                too_long = current_size - segment_size
                too_short = segment_size - (current_size - len(chunk))
                if tolerance >= 0 and min(too_long, too_short) > tolerance:
                    chunk_part0 = chunk[:-too_long]
                    carry = chunk[-too_long:]
                    current_segment[-1] = chunk_part0
                elif too_long >= too_short:
                    carry = current_segment.pop()
                yield current_segment
                current_segment = []
                current_size = 0
    except StopIteration:
        pass
    if current_segment:
        yield current_segment