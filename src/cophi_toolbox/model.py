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
            matrix = pd.SparseDataFrame
        else:
            matrix = pd.DataFrame
        self.dtm = matrix({document.name: utils.count_tokens(document)
                           for document in self.tokens}).T.fillna(0)

    @property
    def size(self):
        return self.dtm.shape

    @property
    def freq_spectrum(self):
        return self.dtm.sum(axis=0).value_counts()

    def get_vocabulary(self):
        return list(self.dtm.columns)

    def get_sorted(self):
        return self.dtm.iloc[:, (-self.dtm.sum()).argsort()]

    def get_mfw(self, n: int = 100):
        return list(self.dtm.iloc[:, :n].columns)

    def get_hl(self):
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm, features: Iterable[str]):
        features = [token for token in features if token in self.model.columns]
        return drop(features, axis=1)

    def get_zscores(self):
        """Calculate z-scores for word frequencies.

        Returns:
            A document-term matrix with z-scores.
        """
        return (self.dtm - self.dtm.mean()) / self.dtm.std()

    def get_rel_freqs(self):
        """Calculate relative word frequencies.

        Returns:
            A document-term matrix with relative word frequencies.
        """
        return self.model.div(self.model.sum(axis=1), axis=0)

    def get_tfidf(self):
        """Calculate TF-IDF.

        Used formula is:

        .. math::
            \mbox{tf-idf}_{t,d} = (1 +\log \mbox{tf}_{t,d}) \cdot \log \frac{N}{\mbox{df}_t}

        Returns:
            A document-term matrix with TF-IDF weighted tokens.
        """
        tf = self.get_rel_freqs()
        idf = np.log(self.size[0] / self.dtm.astype(bool).sum(axis=0))
        return tf * idf

    @property
    def sum_tokens(self):
        return self.dtm.sum(axis=1)

    @property
    def sum_types(self):
        return self.dtm.replace(0, np.nan).count(axis=1)

    def get_ttr(self):
        return self.sum_types / self.sum_tokens

    @property
    def ttr(self):
        return self.sum_types.sum() / self.sum_tokens.sum()

    @property
    def guiraud_r(self):
        """Guiraud (1954)"""
        return self.sum_types.sum() / np.sqrt(self.sum_tokens.sum())

    @property
    def herdan_c(self):
        """Herdan (1960, 1964)"""
        return np.log(self.sum_types.sum()) / np.log(self.sum_tokens.sum())

    @property
    def dugast_k(self):
        """Dugast (1979)"""
        return np.log(self.sum_types.sum()) / np.log(np.log(self.sum_tokens))

    @property
    def maas_a2(self):
        """Maas (1972)"""
        return (np.log(self.sum_tokens) - np.log(self.sum_types.sum())) / (np.log(self.sum_tokens) ** 2)

    @property
    def dugast_u(self):
        """Dugast (1978, 1979)"""
        return (np.log(self.sum_tokens.sum()) ** 2) / (np.log(self.sum_tokens.sum()) - np.log(self.sum_types.sum()))

    @property
    def tuldava_ln(self):
        """Tuldava (1977)"""
        return (1 - (self.sum_types.sum() ** 2)) / ((self.sum_types.sum() ** 2) * np.log(self.sum_tokens.sum()))

    @property
    def brunet_w(self):
        """Brunet (1978)"""
        return self.sum_tokens.sum() ** (self.sum_types.sum() ** 0.172)

    @property
    def cttr(self):
        """Carroll's Corrected Type-Token Ration"""
        return self.sum_types.sum() / np.sqrt(2 * self.sum_tokens.sum())

    @property
    def summer_s(self):
        """Summer's S index"""
        return np.log(np.log(self.sum_types.sum())) / np.log(np.log(self.sum_tokens.sum()))

    @property
    def entropy(self):
        a = -np.log(self.freq_spectrum.index / self.sum_tokens.sum())
        b = self.freq_spectrum / self.sum_tokens.sum()
        return (self.freq_spectrum * a * b).sum()

    @property
    def yule_k(self):
        """Yule (1944)"""
        a = self.freq_spectrum.index / self.sum_tokens.sum()
        b = 1 / self.sum_tokens.sum()
        return 10000 * ((self.freq_spectrum * a ** 2) - b).sum()

    @property
    def simpson_d(self):
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = self.freq_spectrum.index - 1
        return (self.freq_spectrum * a * (b / (self.sum_tokens.sum() - 1))).sum()


    def herdan_vm(text_length, vocabulary_size, frequency_spectrum):
        """Herdan (1955)"""
        a = c.freq_spectrum / c.sum_tokens.sum()
        b = 1 / c.sum_types.sum()
        return np.sqrt(((c.freq_spectrum * a ** 2) - b).sum())

'''
def orlov_z(text_length, vocabulary_size, frequency_spectrum, max_iterations=100, min_tolerance=1):
    """Orlov (1983)
    Approximation via Newton's method.
    """
    def function(text_length, vocabulary_size, p_star, z):
        return (z / math.log(p_star * z)) * (text_length / (text_length - z)) * math.log(text_length / z) - vocabulary_size

    def derivative(text_length, vocabulary_size, p_star, z):
        """Derivative obtained from WolframAlpha:
        https://www.wolframalpha.com/input/?x=0&y=0&i=(x+%2F+(log(p+*+x)))+*+(n+%2F+(n+-+x))+*+log(n+%2F+x)+-+v
        """
        return (text_length * ((z - text_length) * math.log(p_star * z) + math.log(text_length / z) * (text_length * math.log(p_star * z) - text_length + z))) / (((text_length - z) ** 2) * (math.log(p_star * z) ** 2))
    most_frequent = max(frequency_spectrum.keys())
    p_star = most_frequent / text_length
    z = text_length / 2         # our initial guess
    for i in range(def orlov_z(text_length, vocabulary_size, frequency_spectrum, max_iterations=100, min_tolerance=1):
    """Orlov (1983)
    Approximation via Newton's method.
    """
    def function(text_length, vocabulary_size, p_star, z):
        return (z / math.log(p_star * z)) * (text_length / (text_length - z)) * math.log(text_length / z) - vocabulary_size

    def derivative(text_length, vocabulary_size, p_star, z):
        """Derivative obtained from WolframAlpha:
        https://www.wolframalpha.com/input/?x=0&y=0&i=(x+%2F+(log(p+*+x)))+*+(n+%2F+(n+-+x))+*+log(n+%2F+x)+-+v
        """
        return (text_length * ((z - text_length) * math.log(p_star * z) + math.log(text_length / z) * (text_length * math.log(p_star * z) - text_length + z))) / (((text_length - z) ** 2) * (math.log(p_star * z) ** 2))
    most_frequent = max(frequency_spectrum.keys())
    p_star = most_frequent / text_length
    z = text_length / 2         # our initial guess
    for i in range(max_iterations):
        # print(i, text_length, vocabulary_size, p_star, z)
        next_z = z - (function(text_length, vocabulary_size, p_star, z) / derivative(text_length, vocabulary_size, p_star, z))
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    else:
        warnings.warn("Exceeded max_iterations")
return zmax_iterations):
        # print(i, text_length, vocabulary_size, p_star, z)
        next_z = z - (function(text_length, vocabulary_size, p_star, z) / derivative(text_length, vocabulary_size, p_star, z))
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    else:
        warnings.warn("Exceeded max_iterations")
return z
'''