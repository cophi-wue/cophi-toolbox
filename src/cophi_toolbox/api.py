"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the high-level cophi_toolbox API.
"""

from . import model
import pathlib
from typing import Generator, Union, Iterable, Optional
import pandas as pd


def textfile(filepath: str, treat_as: str = ".txt") -> model.Textfile:
    return model.Textfile(filepath, treat_as)


def document(text: str) -> model.Document:
    return model.Document(text)


def corpus(tokens: Iterable[pd.Series]) -> model.Corpus:
    return model.Corpus(tokens)


def pipe(directory: str, suffix: str = ".txt", treat_as: str = ".txt") -> model.Corpus:
    glob = pathlib.Path(directory).glob("**/*{}".format(suffix))

    def lazy_reading(glob):
        for filepath in glob:
            yield textfile(filepath, treat_as)

    documents = pd.Series()
    for document_ in lazy_reading(glob):
        d = document(document_.content)
        tokens = pd.Series(d.tokens)
        tokens.name = document_.name
        documents[document_.name] = tokens
    return corpus(documents)
