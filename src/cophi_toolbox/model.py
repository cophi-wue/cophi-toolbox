"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level corpus model classes to manage and 
process text data in Python.

Complexity measures (:class:`Corpus`) implemented by Thomas Proisl,
see https://github.com/tsproisl/Linguistic_and_Stylistic_Complexity
"""

from . import utils

import logging
import pathlib
import math
import collections
import itertools
from typing import Optional, Iterable, Union, List, Generator, Filter
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
        if not isinstance(self.filepath, pathlib.Path):
            self.filepath = pathlib.Path(self.filepath)
        self.title = self.filepath.stem
        self.suffix = self.filepath.suffix
        self.parent = str(self.filepath.parent)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def read_txt(self) -> str:
        """Read a plain text file.
        """
        return self.filepath.read_text(encoding=self.encoding)

    def parse_xml(self, parser: etree.XMLParser=etree.XMLParser()) -> etree._ElemenTree:
        """Parse a XML file.
        """
        return etree.parse(str(self.filepath), parser=parser)

    @staticmethod
    def stringify(tree) -> str:
        """Stringify an lxml :class:`etree._ElemenTree`.
        """
        return etree.tostring(tree, method="text", encoding=str)

    @property
    def content(self) -> str:
        """Content of textfile.
        """
        logger.debug("Treating '{}' as {}-file ...".format(self.title + self.suffix, self.treat_as))
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
    def unprocessed_tokens(self) -> Generator[str]:
        """Raw, case sensitive (if any) tokens.
        """
        return utils.find_tokens(self.text,
                                 self.pattern,
                                 self.maximum)

    @property
    def tokens(self) -> Generator[str]:
        """Processed, lowered (if any) ngrams.
        """
        tokens = self.unprocessed_tokens
        if self.lowercase:
            tokens = (token.lower() for token in tokens)
        if self.ngrams > 1:
            tokens = utils.get_ngrams(tokens, self.ngrams)
        return tokens

    @property
    def ttr(self) -> float:
        """Type-token ratio.

        Used formula:
        .. math::
            TTR = \frac{V}{N}
        """
        return len(set(self.tokens)) / len(list(self.tokens))

    @staticmethod
    def drop(tokens, features: Iterable[str]) -> Generator[str]:
        """Drop features from tokens.
        """
        return (token for token in tokens if token not in features)

    def get_paragraphs(self, sep: Union[re.compile, str] = re.compile(r"\n")) -> Iterable[str]:
        """Get paragraphs as separate entities.
        """
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        return filter(None, splitted)

    def get_segments(self, segment_size: int = 1000, tolerance: float = 0.05,
                     flatten_chunks: bool = True) -> Iterable[List[str]]:
        """Get segments as separate entities.
        """
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
    documents: Iterable[pd.Series]
    sparse: bool = False

    def __post_init__(self):
        if self.sparse:
            raise NotImplementedError
        else:
            matrix = pd.DataFrame
        self.dtm = matrix({document.name: utils.count_tokens(document)
                           for document in self.documents}).T.fillna(0)

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
    def vocabulary(self) -> List[str]:
        """Corpus vocabulary.
        """
        return list(self.dtm.columns)

    @property
    def sorted_dtm(self) -> pd.DataFrame:
        """Descending sorted document-term matrix.
        """
        return self.dtm.iloc[:, (-self.dtm.sum()).argsort()]

    def get_mfw(self, n: int = 100) -> List[str]:
        """Get the `n` most frequent words.
        """
        return list(self.sorted_dtm.iloc[:, :n].columns)

    def get_hl(self) -> List[str]:
        """Get hapax legomena.
        """
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm, features: Iterable[str]) -> pd.DataFrame:
        """Drop features from document-term matrix.
        """
        features = [token for token in features if token in self.dtm.columns]
        return self.dtm.drop(features, axis=1)

    @property
    def zscores(self) -> pd.DataFrame:
        """Standardized document-term matrix.

        Used formula:
        .. math::
            z_x = \frac{x - \mu}{\sigma}
        """
        return (self.dtm - self.dtm.mean()) / self.dtm.std()

    @property
    def rel_freqs(self) -> pd.DataFrame:
        """Document-term matrix with relative word frequencies.
        """
        return self.dtm.div(self.dtm.sum(axis=1), axis=0)

    @property
    def tfidf(self) -> pd.DataFrame:
        """TF-IDF normalized document-term matrix.

        Used formula:
        .. math::
            \mbox{tf-idf}_{t,d} = (1 +\log \mbox{tf}_{t,d}) \cdot \log \frac{N}{\mbox{idf}_t}
        """
        tf = self.rel_freqs
        idf = self.size["documents"] / self.dtm.astype(bool).sum(axis=0)
        return (1 + np.log(tf)) * np.log(df)

    def get_sum_types(self) -> pd.Series:
        """Get summed type frequencies per document.
        """
        return self.dtm.replace(0, np.nan).count(axis=1)

    def get_sum_tokens(self) -> pd.Series:
        """Get summed token frequencies per document.
        """
        return self.dtm.sum(axis=1)

    @property
    def sum_tokens(self) -> float:
        """Summed token frequencies.
        """
        return self.get_sum_tokens().sum()

    @property
    def sum_types(self) -> float:
        """Summed type frequencies.
        """
        return self.get_sum_types().sum()

    def get_ttr(self) -> pd.Series:
        """Get type-token ratio per document.

        Used formula:
        .. math::
            TTR = \frac{V}{N}
        """
        return self.get_sum_types() / self.get_sum_tokens()

    @property
    def ttr(self) -> float:
        """Type-token ratio.

        Used formula:
        .. math::
            TTR = \frac{V}{N}
        """
        return self.sum_types / self.sum_tokens

    @property
    def guiraud_r(self) -> float:
        """Guiraud's index of lexical richness (1954).

        Used formula:
        .. math::
            r = \frac{V}{\sqrt{N}}
        """
        return self.sum_types / np.sqrt(self.sum_tokens)

    def get_guiraud_r(self) -> pd.Series:
        """Get Guiraud's index of lexical richness (1954) per document.

        Used formula:
        .. math::
            r = \frac{V}{\sqrt{N}}
        """
        return self.get_sum_types() / np.sqrt(self.get_sum_tokens())

    @property
    def herdan_c(self) -> float:
        """Herdan's index of lexical richness (1960, 1964).

        Used formula:
        .. math::
            c = \frac{\log{V}}{\log{N}}
        """
        return np.log(self.sum_types) / np.log(self.sum_tokens)

    def get_herdan_c(self) -> pd.Series:
        """Get Herdan's index of lexical richness (1960, 1964) per document.

        Used formula:
        .. math::
            c = \frac{\log{V}}{\log{N}}
        """
        return np.log(self.get_sum_types()) / np.log(self.get_sum_tokens())

    @property
    def dugast_k(self) -> float:
        """Dugast's uber index (1979).

        Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}
        """
        return np.log(self.sum_types) / np.log(np.log(self.sum_tokens))

    def get_dugast_k(self) -> pd.Series:
        """Get Dugast's uber index (1979) per document.

        Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}
        """
        return np.log(self.get_sum_types()) / np.log(np.log(self.get_sum_tokens()))

    @property
    def maas_a2(self) -> float:
        """Maas' index of lexical richness (1972).
        
        Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}
        """
        return (np.log(self.sum_tokens) - np.log(self.sum_types)) / (np.log(self.sum_tokens) ** 2)

    def get_maas_a2(self) -> pd.Series:
        """Get Maas' index of lexical richness (1972) per document.
        
        Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}
        """
        return (np.log(self.get_sum_tokens()) - np.log(self.get_sum_types())) / (np.log(self.get_sum_tokens()) ** 2)

    @property
    def tuldava_ln(self):
        """Tuldava's index of lexical richness (1977).
        
        Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}
        """
        return (1 - (self.sum_types ** 2)) / ((self.sum_types ** 2) * np.log(self.sum_tokens))

    def get_tuldava_ln(self):
        """Get Tuldava's index of lexical richness (1977) per document.
        
        Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}
        """
        return (1 - (self.get_sum_types() ** 2)) / ((self.get_sum_types() ** 2) * np.log(self.get_sum_tokens()))

    @property
    def brunet_w(self):
        """Get Brunet's index of lexical richness (1978).
        
        Used formula:
        .. math::
            w = V^{V^{0.172}}
        """
        return self.sum_tokens ** (self.sum_types ** 0.172)

    def get_brunet_w(self):
        """Get Brunet's index of lexical richness (1978) per document.
        
        Used formula:
        .. math::
            w = V^{V^{0.172}}
        """
        return self.get_sum_tokens() ** (self.get_sum_types() ** 0.172)

    @property
    def cttr(self):
        """Carroll's corrected type-token ratio.
        
        Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}
        """
        return self.sum_types / np.sqrt(2 * self.sum_tokens)

    def get_cttr(self):
        """Get Carroll's corrected type-token ratio per document.
        
        Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}
        """
        return self.get_sum_types() / np.sqrt(2 * self.get_sum_tokens())

    @property
    def summer_s(self):
        """Summer's index of lexical richness.
        
        Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}
        """
        return np.log(np.log(self.sum_types.sum())) / np.log(np.log(self.sum_tokens.sum()))

    def get_summer_s(self):
        """Get Summer's index of lexical richness per document.
        
        Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}
        """
        return np.log(np.log(self.sum_types)) / np.log(np.log(self.sum_tokens))

    @property
    def entropy(self):
        """Entropy.

        Used formula:
        .. math::
            
        """
        a = -np.log(self.freq_spectrum.index / self.sum_tokens.sum())
        b = self.freq_spectrum / self.sum_tokens.sum()
        return (self.freq_spectrum * a * b).sum()

    @property
    def yule_k(self):
        """Yule (1944).
        
        Used formula (where :math:`N` is the number of tokens, and :math:`V` the number of types):
        .. math::
            \mbox
        """
        a = self.freq_spectrum.index / self.sum_tokens.sum()
        b = 1 / self.sum_tokens.sum()
        return 10000 * ((self.freq_spectrum * a ** 2) - b).sum()

    @property
    def simpson_d(self):
        """Simpson.

        Used formula (where :math:`N` is the number of tokens, and :math:`V` the number of types):
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = self.freq_spectrum.index - 1
        return (self.freq_spectrum * a * (b / (self.sum_tokens.sum() - 1))).sum()

    @property
    def herdan_vm(self):
        """Herdan (1955).

        Used formula (where :math:`N` is the number of tokens, and :math:`V` the number of types):
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = 1 / self.sum_types.sum()
        return np.sqrt(((self.freq_spectrum * a ** 2) - b).sum())