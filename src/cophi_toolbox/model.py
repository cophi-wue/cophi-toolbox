"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level corpus model classes to manage and 
process text data in Python.
"""

from . import utils

import logging
import pathlib
import math
import collections
import itertools
from typing import Optional, Iterable, Union, List
from dataclasses import dataclass, field

from lxml import etree
import pandas as pd
import numpy as np
import regex as re


logger = logging.getLogger("cophi_toolbox.model")


@dataclass
class Textfile:
    filepath: Union[str, pathlib.Path]
    treat_as: str = ".txt"
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.treat_as not in {".txt", ".xml"}:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(self.treat_as))
        if isinstance(self.filepath, str):
            self.filepath = pathlib.Path(self.filepath)
        self.title = self.filepath.stem
        self.suffix = self.filepath.suffix
        self.parent = str(self.filepath.parent)

    def __enter__(self):
        return self.content

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def read_txt(self):
        return self.filepath.read_text(encoding=self.encoding)

    def parse_xml(self, parser: etree.XMLParser=etree.XMLParser()):
        return etree.parse(str(self.filepath), parser=parser)

    @staticmethod
    def stringify(tree):
        return etree.tostring(tree, method="text", encoding=str)

    @property
    def content(self):
        if self.treat_as == ".txt":
            return self.read_txt()
        elif self.treat_as == ".xml":
            tree = self.parse_xml()
            return self.stringify(tree)


@dataclass
class Document:
    text: str
    lowercase: bool = True
    ngrams: int = 1
    pattern: str = r"\p{L}+\p{P}?\p{L}+"
    maximum: Optional[int] = None

    @property
    def unprocessed_tokens(self):
        return utils.find_tokens(self.text,
                                 self.pattern,
                                 self.maximum)

    @property
    def tokens(self):
        tokens = self.unprocessed_tokens
        if self.lowercase:
            tokens = (token.lower() for token in tokens)
        if self.ngrams > 1:
            tokens = utils.get_ngrams(tokens, self.ngrams)
        return tokens

    @staticmethod
    def drop(tokens, features: List[str]):
        return (token for token in tokens if token not in features)

    def get_paragraphs(self, sep: Union[re.compile, str] = re.compile(r"\n")) -> Iterable[str]:
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        return filter(None, splitted)

    def get_segments(self, segment_size: int = 1000, tolerance: float = 0.05,
                     flatten_chunks: bool = True) -> Iterable[List[str]]:
        segments = utils.segment_fuzzy([self.tokens],
                                       segment_size,
                                       tolerance)
        if flatten_chunks:
            if not callable(flatten_chunks):
                def flatten_chunks(segment):
                    return list(itertools.chain.from_iterable(segment))
            segments = map(flatten_chunks, segments)
        return segments


@dataclass
class Corpus(pd.DataFrame):
    tokens: Optional[Iterable[pd.Series]] = None

    @property
    def size(self):
        return len(self.tokens) if self.tokens is not None else 0

    def dtm(self, sparse: bool = False):
        """Create classic document-term matrix, construct model object.

        Parameters:
            dense: If True, create dense document-term matrix.
        """
        self.model = pd.DataFrame({document.name: utils.count_tokens(document)
                                   for document in self.tokens}).T.fillna(0)
        if sparse:
            self.model = self.model.to_sparse()

    def sort(self, ascending: bool = False):
        """Sort corpus model by frequency.
        """
        self.model = self.model.loc[:, self.model.sum().sort_values(ascending=ascending).index]

    def mfw(self, threshold: int = 100):
        """Get the most frequent words from corpus object.
        """
        self.sort()
        self.mfw = list(self.model.iloc[:, :threshold].columns)

    @property
    def hl(self):
        """Get hapax legomena from corpus object.

        Returns:
            Hapax legomena of the corpus.
        """
        return list(self.model.loc[:, self.model.max() == 1].columns)

    def drop(self, features: Iterable[str]):
        """Drop features (tokens, or words) from model object.

        Parameters:
            features: Tokens to remove from the tokens object.
        """
        features = [token for token in features if token in self.model.columns]
        self.model = self.model.drop(features, axis=1)

    @property
    def zscores(self):
        """Calculate z-scores for word frequencies.

        Returns:
            A document-term matrix with z-scores.
        """
        return (self.model - self.model.mean()) / self.model.std()

    @property
    def rel_freqs(self):
        """Calculate relative word frequencies.

        Returns:
            A document-term matrix with relative word frequencies.
        """
        return self.model.div(self.model.sum(axis=1), axis=0)

    @property
    def tfidf(self):
        """Calculate TF-IDF.

        Used formula is:

        .. math::
            \mbox{tf-idf}_{t,d} = (1 +\log \mbox{tf}_{t,d}) \cdot \log \frac{N}{\mbox{df}_t}

        Returns:
            A document-term matrix with TF-IDF weighted tokens.
        """
        tf = self.rel_freqs
        idf = np.log(self.size / self.model.astype(bool).sum(axis=0))
        return tf * idf

    def to_disk(self, filepath: str, **kwargs: str):
        """Write corpus model object to disk.

        Parameters:
            filepath: Path to text file.
            **kwargs: See :func:`pd.DataFrame().to_csv()`.
        """
        self.model.to_csv(filepath, **kwargs)

    def from_disk(self, filepath: str, **kwargs: str):
        """Read corpus model object from disk.

        Parameters:
            filepath: Path to text file.
            **kwargs: See :func:`pd.read_csv()`.
        """
        self.model = pd.read_csv(filepath, index_col=0, **kwargs)