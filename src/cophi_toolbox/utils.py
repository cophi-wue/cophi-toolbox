from typing import Iterable
import pandas as pd
import collections
import regex as re


def get_ngrams(tokens: Iterable[str], n: int, sep: str = " ") -> Iterable[str]:
    """Constructs ngrams.

    Parameters:
        tokens:
        n:
        sep:
    
    Returns:

    """
    return (sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)]))

def count_tokens(tokens: Iterable[str]) -> pd.Series:
    """Counts tokens.

    Parameters:
        tokens:
    
    Returns:
    """
    return pd.Series(collections.Counter(tokens))

def find_tokens(document, pattern, maximum):
    count = 0
    for match in re.compile(pattern).finditer(document):
        count += 1
        yield match.group(0)
        if maximum is not None and count >= maximum:
            return

def segment_fuzzy(paragraphs, segment_size=5000, tolerance=0.05):
    if tolerance > 0 and tolerance < 1:
        tolerance = round(segment_size * tolerance)

    current_segment = []
    current_size = 0
    carry = None
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