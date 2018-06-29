"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the cophi_toolbox API.
"""

from . import model
from typing import Generator, Union


def token(document: str, **kwargs: str) -> model.Token:
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
    """Represents a document on the document level.

    Parameters:
        filepath: Path to text file.
        treat_as: Treat a file like one with this suffix.
        encoding: Encoding to use for UTF when reading.

    Returns:
        A Document object.

    Example:
        >>> d = document("corpus/goethe_werther.txt")
        >>> d.name
        "goethe_werther"
        >>> d.document
        "Wie froh bin ich, ..."
        >>> d.paragraphs
        ['Wie froh bin ich, ...', ...]
        >>> d.segments
        ['Wie froh bin ich, ...', ...]
    """
    d = model.Document(filepath, **kwargs)
    d.from_disk()
    d.split_paragraphs()
    d.segment()
    return d

def corpus(tokens: Iterable[pd.Series[str]]) -> model.Corpus:
    """Represents a corpus on the corpus level.

    Parameters:
        tokens: 

    Returns:
        A Corpus object.

    Example:
        >>> c = corpus([["everything's", 'gone', 'green']])
        >>> c.model

        >>> c.mfw(1)

        >>> c.hl

    """
    c = model.Corpus(tokenized_documents)
    c.document_term_matrix()
    c.get_mfw()
    c.get_hapax_legomena()
    return c