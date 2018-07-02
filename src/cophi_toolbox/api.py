"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the high-level cophi_toolbox API.
"""

import pathlib
from . import model
from typing import Generator, Union, Iterable, Optional
import pandas as pd


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
        >>> d.text
        "Everything's gone green."
    """
    d = model.Document(filepath, **kwargs)
    d.from_disk()
    d.get_paragraphs()
    d.get_segments()
    return d

def token(text: str, **kwargs: str) -> model.Token:
    """Represent a document on the token level.

    Parameters:
        text: String to be tokenized.
        pattern: Regular expression token pattern.
        maximum: If set, stop reading after that many words.
        lowercase: If True, normalize all tokens to lowercase.
        ngrams: Ngram size.

    Returns:
        A Token object.

    Example:
        >>> t = token("Everything's gone green.")
        >>> t.text
        "Everything's gone green."
        >>> list(t.tokens)
        ["everything's", 'gone', 'green']
    """
    t = model.Token(text, **kwargs)
    t.tokenize()
    t.postprocess()
    return t

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
        >>> c.model  # doctest: +NORMALIZE_WHITESPACE
             everything's  gone  green
        doc             1     1      2
        >>> c.mfw(threshold=1)  # defaults to 100
        >>> list(c.mfw)  # most frequent words
        ["green"]
        >>> list(c.hl)  # hapax legomena
        ["everything's", "gone"]
    """
    c = model.Corpus(tokens)
    c.dtm()
    c.get_mfw()
    c.get_hl()
    return c

def pipe(directory: str, encoding: str = "utf-8", suffix: str = ".txt",
         treat_as: str = ".txt", pattern: str = r"\p{L}+\p{P}?\p{L}+",
         maximum: Optional[int] = None, lowercase: bool = True,
         ngrams: int = 1) -> model.Corpus:
         """Pipe files in a directory to the corpus model.

         Parameters:
            directory:
            encoding: Encoding to use for UTF when reading.
            suffix:
            treat_as: Treat a file like one with this suffix.
            pattern: Regular expression token pattern.
            maximum: If set, stop reading after that many words.
            lowercase: If True, normalize all tokens to lowercase.
            ngrams: Ngram size.

        Returns:
            A Corpus model object.
         """
    glob = pathlib.Path(directory).glob("**/*{}".format(suffix))
    def lazy_reading(glob):
        for filepath in glob:
            d = model.Document(filepath,
                               encoding=encoding,
                               treat_as=treat_as)
            d.from_disk()
            yield d

    corpus = pd.Series()
    for document in lazy_reading(glob):
        t = model.Token(document.text,
                        pattern=pattern,
                        maximum=maximum,
                        lowercase=lowercase,
                        ngrams=ngrams)
        t.tokenize()
        t.postprocess()
        tokens = pd.Series(t.tokens)
        tokens.name = document.name
        corpus[document.name] = tokens
    return model.Corpus(corpus)

