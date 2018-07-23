"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level corpus model classes to manage and 
process text data in Python.

Complexity measures (:class:`Corpus`) implemented by Thomas Proisl,
see https://github.com/tsproisl/Linguistic_and_Stylistic_Complexity
"""

#from . import utils

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
        return self

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
        logger.info("Processing '{}' as '{}' ...".format(self.title, self.treat_as))
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

    @property
    def ttr(self):
        return len(set(self.tokens)) / len(self.tokens)

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
class Corpus:
    tokens: Iterable[pd.Series]
    sparse: bool = False

    def __post_init__(self):
        if self.sparse:
            raise NotImplementedError
        else:
            matrix = pd.DataFrame
        self.dtm = matrix({document.name: utils.count_tokens(document)
                           for document in self.tokens}).T.fillna(0)

    @property
    def size(self) -> pd.Series:
        """Number of documents and types.
        """
        return pd.Series(self.dtm.shape, index=["documents", "types"])

    @property
    def freq_spectrum(self) -> pd.Series:
        """Frequency spectrum of types.
        """
        return self.dtm.sum(axis=0).value_counts()

    @property
    def vocabulary(self) -> List:
        """Corpus vocabulary.
        """
        return list(self.dtm.columns)

    @property
    def sorted_dtm(self) -> pd.DataFrame:
        """Descending sorted document-term matrix.
        """
        return self.dtm.iloc[:, (-self.dtm.sum()).argsort()]

    def get_mfw(self, n: int = 100):
        """Get the `n` most frequent words from corpus.
        """
        return list(self.sorted_dtm.iloc[:, :n].columns)

    def get_hl(self):
        """Get hapax legomena from corpus.
        """
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm, features: Iterable[str]):
        """Drop words (or, `features`) from document-term matrix.
        """
        features = [token for token in features if token in dtm.columns]
        return dtm.drop(features, axis=1)

    @property
    def zscores(self):
        """Standardized document-term matrix.

        Used formula is:
        .. math::
            \mbox
        """
        return (self.dtm - self.dtm.mean()) / self.dtm.std()

    @property
    def rel_freqs(self):
        """Document-term matrix with relative word frequencies.
        """
        return self.dtm.div(self.dtm.sum(axis=1), axis=0)

    @property
    def tfidf(self):
        """TF-IDF normalized document-term matrix.

        Used formula is:
        .. math::
            \mbox{tf-idf}_{t,d} = (1 +\log \mbox{tf}_{t,d}) \cdot \log \frac{N}{\mbox{df}_t}
        """
        tf = self.rel_freqs
        idf = np.log(self.size["documents"] / self.dtm.astype(bool).sum(axis=0))
        return tf * idf

    @property
    def sum_tokens(self):
        """Summed token frequencies.
        """
        return self.dtm.sum(axis=1)

    @property
    def sum_types(self):
        """Summed type frequencies.
        """
        return self.dtm.replace(0, np.nan).count(axis=1)

    def get_ttr(self):
        """Get type-token ratio per document.

        Used formula is:
        .. math::
            \mbox{ttr} = 
        """
        return self.sum_types / self.sum_tokens

    @property
    def ttr(self):
        """Type-token ratio.

        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_types.sum() / self.sum_tokens.sum()

    @property
    def guiraud_r(self):
        """Guiraud (1954).

        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_types.sum() / np.sqrt(self.sum_tokens.sum())

    def get_guiraud_r(self):
        """Get Guiraud (1954) per document.

        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_types / np.sqrt(self.sum_tokens)

    @property
    def herdan_c(self):
        """Herdan (1960, 1964).
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(self.sum_types.sum()) / np.log(self.sum_tokens.sum())

    def get_herdan_c(self):
        """Get Herdan (1960, 1964) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(self.sum_types) / np.log(self.sum_tokens)

    @property
    def dugast_k(self):
        """Dugast (1979).
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(self.sum_types.sum()) / np.log(np.log(self.sum_tokens.sum()))

    def get_dugast_k(self):
        """Get Dugast (1979) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(self.sum_types) / np.log(np.log(self.sum_tokens))

    @property
    def maas_a2(self):
        """Maas (1972).
        
        Used formula is:
        .. math::
            \mbox
        """
        return (np.log(self.sum_tokens.sum()) - np.log(self.sum_types.sum())) / (np.log(self.sum_tokens.sum()) ** 2)

    def get_maas_a2(self):
        """Get Maas (1972) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return (np.log(self.sum_tokens) - np.log(self.sum_types)) / (np.log(self.sum_tokens) ** 2)

    @property
    def dugast_u(self):
        """Dugast (1978, 1979).
        
        Used formula is:
        .. math::
            \mbox
        """
        return (np.log(self.sum_tokens.sum()) ** 2) / (np.log(self.sum_tokens.sum()) - np.log(self.sum_types.sum()))

    def get_dugast_u(self):
        """Get Dugast (1978, 1979) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return (np.log(self.sum_tokens) ** 2) / (np.log(self.sum_tokens) - np.log(self.sum_types))

    @property
    def tuldava_ln(self):
        """Tuldava (1977).
        
        Used formula is:
        .. math::
            \mbox
        """
        return (1 - (self.sum_types.sum() ** 2)) / ((self.sum_types.sum() ** 2) * np.log(self.sum_tokens.sum()))

    def get_tuldava_ln(self):
        """Get Tuldava (1977) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return (1 - (self.sum_types ** 2)) / ((self.sum_types ** 2) * np.log(self.sum_tokens))

    @property
    def brunet_w(self):
        """Brunet (1978).
        
        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_tokens.sum() ** (self.sum_types.sum() ** 0.172)

    def get_brunet_w(self):
        """Get Brunet (1978) per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_tokens ** (self.sum_types ** 0.172)

    @property
    def cttr(self):
        """Carroll's Corrected Type-Token Ration.
        
        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_types.sum() / np.sqrt(2 * self.sum_tokens.sum())

    def get_cttr(self):
        """Get Carroll's Corrected Type-Token Ration per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return self.sum_types / np.sqrt(2 * self.sum_tokens)

    @property
    def summer_s(self):
        """Summer's S index.
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(np.log(self.sum_types.sum())) / np.log(np.log(self.sum_tokens.sum()))

    def get_summer_s(self):
        """Get Summer's S index per document.
        
        Used formula is:
        .. math::
            \mbox
        """
        return np.log(np.log(self.sum_types)) / np.log(np.log(self.sum_tokens))

    @property
    def entropy(self):
        """Entropy.

        Used formula is:
        .. math::
            \mbox
        """
        a = -np.log(self.freq_spectrum.index / self.sum_tokens.sum())
        b = self.freq_spectrum / self.sum_tokens.sum()
        return (self.freq_spectrum * a * b).sum()

    @property
    def yule_k(self):
        """Yule (1944).
        
        Used formula is:
        .. math::
            \mbox
        """
        a = self.freq_spectrum.index / self.sum_tokens.sum()
        b = 1 / self.sum_tokens.sum()
        return 10000 * ((self.freq_spectrum * a ** 2) - b).sum()

    @property
    def simpson_d(self):
        """Simpson.

        Used formula is:
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = self.freq_spectrum.index - 1
        return (self.freq_spectrum * a * (b / (self.sum_tokens.sum() - 1))).sum()

    @property
    def herdan_vm(self):
        """Herdan (1955).

        Used formula is:
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = 1 / self.sum_types.sum()
        return np.sqrt(((self.freq_spectrum * a ** 2) - b).sum())

    def get_orlov_z(self, iterations=100, min_tolerance=1):
        """Orlov (1983).
        Approximation via Newton's method.
        """
        total = self.sum_tokens.sum()
        most_frequent = self.freq_spectrum.max()
        p_star = most_frequent / total
        z = total / 2
        for i in range(iterations):
            a = (z / np.log(p_star * z)) * (total / (total - z)) * np.log(total / z) - self.sum_types.sum()
            b = (total * ((z - total) * np.log(p_star * z) + np.log(total / z) * (total * np.log(p_star * z) - total + z))) / (((total - z) ** 2) * (np.log(p_star * z) ** 2))
            next_z = z - (a / b)
            abs_diff = abs(z - next_z)
            z = next_z
            if abs_diff <= min_tolerance:
                break
        return z