"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the high-level cophi_toolbox API.
"""

from . import model
from typing import Generator, Union, Iterable
import pandas as pd


def token(document: str, **kwargs: str) -> model.Token:
    """Represent a document on the token level.

    Parameters:
        document: String to be tokenized.
        pattern: Regular expression token pattern.
        maximum: If set, stop reading after that many words.
        lowercase: If True, normalize all tokens to lowercase.
        ngrams: Ngram size.

    Returns:
        A Token object.

    Example:
        >>> t = token("Everything's gone green.")
        >>> t.document
        "Everything's gone green."
        >>> list(t.tokens)
        ["everything's", 'gone', 'green']
    """
    t = model.Token(document, **kwargs)
    t.tokenize()
    t.postprocess()
    return t


def document(filepath: str, **kwargs: str) -> model.Document:
    """Represent a document on the document level.

    Parameters:
        filepath: Path to text file.
        treat_as: Treat a file like one with this suffix.
        encoding: Encoding to use for UTF when reading.

    Returns:
        A Document object.

    Example:
        >>> d = document("corpus/document.txt")
        >>> d.name
        "document"
        >>> d.document
        "Everything's gone green."
    """
    d = model.Document(filepath, **kwargs)
    d.from_disk()
    d.get_paragraphs()
    d.get_segments()
    return d

def corpus(tokens: Iterable[pd.Series]) -> model.Corpus:
    """Represent a corpus on the corpus level.

    Parameters:
        tokens: Tokenized corpus.

    Returns:
        A Corpus object.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series(["everything's", "gone", "green", "green"], name="doc")
        >>> c = corpus([s])
        >>> c.model  # +NORMALIZE_WHITESPACE
             everything's  gone  green
        doc             1     1      2
        >>> c.mfw(threshold=1)  # defaults to 100
        >>> list(c.mfw)
        ["green"]
        >>> list(c.hl)
        ["everything's", "gone"]

    """
    c = model.Corpus(tokens)
    c.dtm()
    c.get_mfw()
    c.get_hl()
    return c