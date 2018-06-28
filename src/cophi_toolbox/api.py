"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the cophi_toolbox API.
"""

import model
from typing import Generator


def tokens(document: str, **kwargs):
    """Represents a document on the token level.

    Parameters:
        document: String to be tokenized.
        pattern: Regular expression token pattern.
        maximum: If set, stop reading after that many words.
        lowercase: If True, normalize all tokens to lowercase.
        ngrams: Ngram size.

    Returns:
        A Token object.

    Example:
        >>> t = tokens("Everything's gone green.")
        >>> t.document
        "Everything's gone green."
        >>> t.tokens
        ["everything's", 'gone', 'green']
    """
    t = model.Token(document, **kwargs)
    t.tokenize()
    t.postprocess()
    return t


def document(filepath: str, **kwargs):
    """Represents a document on the document level.

    Parameters:
        filepath: Path to text file.
        treat_as: Treat a file like one with this suffix.
        encoding: Encoding to use for UTF when reading.

    Returns:
        A Document object.

    Example:
        >>> d = document("~/corpus/goethe_werther.txt")
        >>> d.name
        "goethe_werther"
        >>> d.text
        "Wie froh bin ich, ..."
        >>> d.paragraphs
        blabla
        >>> d.segments
        blabla
    """
    d = model.Document(filepath, **kwargs)
    d.read()
    d.split_paragraphs()
    d.segment()
    return d

def corpus(files, **kwargs):
    """Represents a corpus on the corpus level.

    Parameters:
        files:
    """
    c = model.Corpus(**kwargs)
    c.process(files)
    return c