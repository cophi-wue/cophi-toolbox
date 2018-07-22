"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the high-level cophi_toolbox API.
"""

from . import model

import pathlib
from typing import Generator, Union, Iterable, Optional

import pandas as pd


def textfile(filepath: str, **kwargs: str) -> model.Textfile:
    return model.Textfile(filepath, **kwargs)


def document(text: str, **kwargs: str) -> model.Document:
    return model.Document(text, **kwargs)


def corpus(tokens: Iterable[pd.Series]) -> model.Corpus:
    """Represent a corpus on the corpus level.

    Parameters:
        tokens: Tokenized corpus.

    Returns:
        A Corpus object.
    """
    c = model.Corpus(tokens)
    c.dtm()
    c.mfw()
    return c

def pipe(directory: str, encoding: str = "utf-8", suffix: str = ".txt",
         treat_as: str = ".txt", pattern: str = r"\p{L}+\p{P}?\p{L}+",
         maximum: Optional[int] = None, lowercase: bool = True,
         ngrams: int = 1) -> model.Corpus:
    """Pipe files in a directory to the corpus model.

    Parameters:
        directory: Path to corpus directory.
        encoding: Encoding to use for UTF when reading.
        suffix: Suffix of text files.
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
            yield textfile(filepath, encoding=encoding, treat_as=treat_as)

    documents = pd.Series()
    for document_ in lazy_reading(glob):
        d = document(document_.text, pattern=pattern, maximum=maximum,
                     lowercase=lowercase, ngrams=ngrams)
        tokens = pd.Series(d.tokens)
        tokens.name = document_.name
        documents[document_.name] = tokens
    return corpus(documents)
