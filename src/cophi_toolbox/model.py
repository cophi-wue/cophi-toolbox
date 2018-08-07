"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level model classes to manage and 
process text data in Python.
"""

from . import utils

import logging
import pathlib
import collections
import itertools
from typing import Optional, Iterable, Union, List, Generator
from dataclasses import dataclass

from lxml import etree
import pandas as pd
import numpy as np
import regex as re


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

    def read_txt(self) -> str:
        """Read a plain text file.
        """
        return self.filepath.read_text(encoding=self.encoding)

    def parse_xml(self, parser: etree.XMLParser=etree.XMLParser()) -> etree._ElementTree:
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
    def unprocessed_tokens(self) -> Generator[str, None, None]:
        """Raw, case sensitive (if any) tokens.
        """
        return utils.find_tokens(self.text,
                                 self.pattern,
                                 self.maximum)

    @property
    def tokens(self) -> Generator[str, None, None]:
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
    def drop(tokens, features: Iterable[str]) -> Generator[str, None, None]:
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

    @staticmethod
    def map_metadata(matrix: Union[pd.DataFrame, pd.Series], metadata: pd.DataFrame, uuid: str = "uuid",
                     fields: Union[str, List[str]] = ["title"], sep: str = "_") -> pd.DataFrame:
        if isinstance(fields, str):
            fields = [fields]
        matrix = matrix.copy()  # do not work on original object itself
        document_id = metadata[uuid]
        ix = metadata[fields[0]].astype(str)
        if len(fields) > 1:
            for field in fields[1:]:
                ix = ix + sep + metadata[field].astype(str)
        document_id.index = ix
        matrix.index = document_id.to_dict()
        return matrix

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
    def types(self) -> List[str]:
        """Corpus vocabulary.
        """
        return list(self.dtm.columns)

    @staticmethod
    def sort(dtm) -> pd.DataFrame:
        """Descending sorted document-term matrix.
        """
        return dtm.iloc[:, (-dtm.sum()).argsort()]

    def mfw(self, n: int = 100, rel_freqs=True) -> List[str]:
        """Get the `n` most frequent words.
        """
        if rel_freqs:
            dtm = self.rel_freqs
        else:
            dtm = self.dtm
        return list(self.sort(dtm).iloc[:, :n].columns)

    def hapax(self) -> List[str]:
        """Get hapax legomena.
        """
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm, features: Iterable[str]) -> pd.DataFrame:
        """Drop features from document-term matrix.
        """
        features = [token for token in features if token in dtm.columns]
        return dtm.drop(features, axis=1)

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
            tf-idf_{t,d} \; = \; tf_{t,d} \times idf_t \; = \; tf_{t,d} \times log(\frac{N}{df_t})
        """
        tf = self.rel_freqs
        idf = self.size["documents"] / self.dtm.astype(bool).sum(axis=0)
        return tf * np.log(idf)

    def sum_types(self, window=None) -> pd.Series:
        """Summed type frequencies per document.
        """
        if window:
            w = self.dtm.copy()  # do not work on original object itself
            w = w.replace(0, np.nan).T.rolling(window).count().T
            w.columns = ["window_{}".format(n) for n in range(self.size["types"])]
            return w
        else:
            return self.dtm.replace(0, np.nan).count(axis=1)

    def sum_tokens(self, window=None) -> pd.Series:
        """Summed token frequencies per document.
        """
        if window:
            w = self.dtm.copy()  # do not work on original object itself
            w = w.rolling(window, min_periods=1, axis=1).sum()
            w.columns = ["window_{}".format(n) for n in range(self.size["types"])]
            return w
        else:
            return self.dtm.sum(axis=1)

    def complexity(self, measure="ttr", window=1000):
        if measure == "ttr":
            if window:
                sttr = self.sum_types(window) / self.sum_tokens(window)
                return sttr.mean(axis=1)
            else:
                logging.warning("It is not a good idea to compare unstandardized TTR values "
                                "of several texts, because the TTR is strongly influenced by "
                                "text length. By the way, the standard deviation of the text "
                                "lengths in your corpus is {}, and it would be 0 if all texts "
                                "were of equal length. To calculate the standardized TTR, define "
                                "a value for `window`, so that a window of `n` tokens slides over "
                                "the text and calculates multiple TTRs, from which the mean value "
                                "is taken at the end.".format(self.sum_tokens().std()))
                return self.sum_types() / self.sum_tokens()
        elif measure == "guiraud_r":
            if window is not None:
                guiraud_r = self.sum_types(window) / np.sqrt(self.sum_tokens(window))
                return guiraud_r.mean(axis=1)
            else:
                return self.sum_types() / np.sqrt(self.sum_tokens())
        elif measure == "herdan_c":
            return np.log(self.sum_types) / np.log(self.sum_tokens)
        elif measure == "dugast_k":
            return np.log(self.sum_types) / np.log(np.log(self.sum_tokens))
        elif measure == "maas_a2":
            return (np.log(self.sum_tokens) - np.log(self.sum_types)) / (np.log(self.sum_tokens) ** 2)
        elif measure == "tuldava_ln":
            return (1 - (self.sum_types ** 2)) / ((self.sum_types ** 2) * np.log(self.sum_tokens))
        elif measure == "brunet_w":
            return self.sum_tokens ** (self.sum_types ** 0.172)
        elif measure == "cttr":
            return self.sum_types / np.sqrt(2 * self.sum_tokens)
        elif measure == "summers_s":
            return np.log(np.log(self.sum_types) / np.log(np.log(self.sum_tokens)))
        else:
            raise NotImplementedError("The measure '{}' is not implemented.".format(measure))

    @property
    def sichel_s(self):
        """Sichel (1975)"""
        return self.freq_spectrum[2] / self.sum_types.sum()

    @property
    def michea_m(self):
        """Michéa (1969, 1971)"""
        return self.sum_types.sum() / self.freq_spectrum[2]

    @property
    def honore_h(self):
        """Honoré (1979)"""
        return 100 * (np.log(self.sum_tokens.sum()) / (1 - (self.freq_spectrum[1] / self.sum_types.sum())))

    @property
    def ttr(self) -> float:
        """Type-token ratio.

        Used formula:
        .. math::
            TTR = \frac{V}{N}
        """
        return self.sum_types.sum() / self.sum_tokens.sum()

    @property
    def guiraud_r(self) -> float:
        """Guiraud's index of lexical richness (1954).

        Used formula:
        .. math::
            r = \frac{V}{\sqrt{N}}
        """
        return self.sum_types.sum() / np.sqrt(self.sum_tokens.sum())

    @property
    def herdan_c(self) -> float:
        """Herdan's index of lexical richness (1960, 1964).

        Used formula:
        .. math::
            c = \frac{\log{V}}{\log{N}}
        """
        return np.log(self.sum_types) / np.log(self.sum_tokens.sum())

    @property
    def dugast_k(self) -> float:
        """Dugast's uber index (1979).

        Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}
        """
        return np.log(self.sum_types.sum()) / np.log(np.log(self.sum_tokens.sum()))

    @property
    def maas_a2(self) -> float:
        """Maas' index of lexical richness (1972).
        
        Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}
        """
        return (np.log(self.sum_tokens.sum()) - np.log(self.sum_types.sum())) / (np.log(self.sum_tokens.sum()) ** 2)

    @property
    def tuldava_ln(self):
        """Tuldava's index of lexical richness (1977).
        
        Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}
        """
        return (1 - (self.sum_types.sum() ** 2)) / ((self.sum_types.sum() ** 2) * np.log(self.sum_tokens.sum()))

    @property
    def brunet_w(self):
        """Get Brunet's index of lexical richness (1978).
        
        Used formula:
        .. math::
            w = V^{V^{0.172}}
        """
        return self.sum_tokens.sum() ** (self.sum_types.sum() ** 0.172)

    @property
    def cttr(self):
        """Carroll's corrected type-token ratio.
        
        Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}
        """
        return self.sum_types.sum() / np.sqrt(2 * self.sum_tokens.sum())

    @property
    def summer_s(self):
        """Summer's index of lexical richness.
        
        Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}
        """
        return np.log(np.log(self.sum_types.sum())) / np.log(np.log(self.sum_tokens.sum()))

    @property
    def entropy(self):
        """Entropy.

        Used formula:
        .. math::
            https://docs.quanteda.io/reference/textstat_lexdiv.html
        """
        a = -np.log(self.freq_spectrum.index / self.sum_tokens.sum())
        b = self.freq_spectrum / self.sum_tokens.sum()
        return (self.freq_spectrum * a * b).sum()

    @property
    def yule_k(self):
        """Yule (1944).
        
        Used formula:
        .. math::
            K = 10^4 \times \frac{(\sum_{X=1}^{X}{{f_X}X^2}) - N}{N^2}
        """
        a = self.freq_spectrum.index / self.sum_tokens.sum()
        b = 1 / self.sum_tokens.sum()
        return 10 ** 4 * ((self.freq_spectrum * a ** 2) - b).sum()

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
        b = 1 / self.sum_types.sum().sum()
        return np.sqrt(((self.freq_spectrum * a ** 2) - b).sum())