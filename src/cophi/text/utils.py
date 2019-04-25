"""
cophi.text.utils
~~~~~~~~~~~

This module implements low-level helper functions.
"""

import collections
import itertools

import pandas as pd
import regex as re

from cophi.text import model


def construct_ngrams(tokens, n=2, sep=" "):
    """
    Parameters:
        tokens (list): The tokenized document.
        n (int): Number of tokens per ngram.
        sep (str): Separator between tokens within an ngram.
    """
    return (sep.join(ngram)
            for ngram in zip(*(itertools.islice(i, token, None)
                             for token, i in enumerate(itertools.tee(tokens,
                                                                     n)))))


def find_tokens(document,
                token_pattern=r"\p{L}+\p{Connector_Punctuation}?\p{L}+",
                maximum=None):
    """
    Parameters:
        document (str): The text of a document.
        token_pattern (str): Regex pattern for one token.
        maximum (int): Stop tokenizing after that much tokens.
    """
    count = 0
    for match in re.compile(token_pattern).finditer(document):
        count += 1
        yield match.group(0)
        if maximum is not None and count >= maximum:
            return


def lowercase_tokens(tokens):
    """
    Parameters:
        tokens (list): The tokenized document.
    """
    return [token.lower() for token in tokens]


def segment_fuzzy(paragraphs, size=1000, tolerance=0.05):
    """Segment a string, respecting paragraphs.

    Parameters:
        paragraphs: Paragraphs of a document as separated entities.
        size: The target length of each segment in tokens.
        tolerance: How much may the actual segment size differ from the `size`?
            If ``0 < tolerance < 1``, this is interpreted as a fraction
            of the `size`, otherwise it is interpreted as an absolute number.
            If ``tolerance < 0``, paragraphs are never split apart.
    """
    if tolerance > 0 and tolerance < 1:
        tolerance = round(size * tolerance)
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
            if current_size >= size:
                too_long = current_size - size
                too_short = size - (current_size - len(chunk))
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


def _parameter(tokens, measure):
    """Count types, tokens and occuring frequencies.

    Parameters:
        tokens (list): The tokenized document.
        measure (str): Complexity measure you need the results to.
    """
    if measure in {"sichel_s", "michea_m"}:
        bow = collections.Counter(tokens)
        freq_spectrum = collections.Counter(bow.values())
        return {"num_types": len(bow), "freq_spectrum": freq_spectrum}
    elif measure in {"honore_h", "herdan_vm", "orlov_z"}:
        bow = collections.Counter(tokens)
        freq_spectrum = collections.Counter(bow.values())
        return {"num_types": len(bow), "num_tokens": len(tokens),
                "freq_spectrum": pd.Series(freq_spectrum)}
    elif measure in {"entropy", "yule_k", "simpson_d"}:
        bow = collections.Counter(tokens)
        freq_spectrum = collections.Counter(bow.values())
        return {"num_tokens": len(tokens),
                "freq_spectrum": pd.Series(freq_spectrum)}
    else:
        return {"num_types": len(set(tokens)), "num_tokens": len(tokens)}


def export(dtm, filepath, format="text"):
    """Export a document-term matrix.

    Parameters:
        dtm: A document-term matrix.
        filepath: Path to output file. Possible values are `plaintext`/`text` or
            `svmlight`.
        format: File format.
    """
    if format.lower() in {"plaintext", "text"}:
        model.Corpus.plaintext(dtm, filepath)
    elif format.lower() in {"svmlight"}:
        model.Corpus.svmlight(dtm, filepath)
    else:
        raise ValueError("'{}' is no supported file format.".format(format))
