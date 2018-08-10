"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level model classes to manage and 
process text data in Python.
"""

from . import utils
from . import complexity

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
    treat_as: Optional[str] = None
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.treat_as and self.treat_as not in {".txt", ".xml"}:
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
        """Parse an XML file.
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
        if (not self.treat_as and self.suffix == ".txt") or self.treat_as == ".txt":
            return self.read_txt()
        elif (not self.treat_as and self.suffix == ".xml") or self.treat_as == ".xml":
            tree = self.parse_xml()
            return self.stringify(tree)

    @property
    def size(self):
        """Size of document in characters.
        """
        return len(self.content)

@dataclass
class Document:
    text: str
    title: Optional[str] = None
    lowercase: bool = True
    ngrams: int = 1
    pattern: str = r"\p{L}+\p{P}?\p{L}+"
    maximum: Optional[int] = None

    @property
    def unprocessed_tokens(self) -> Generator[str, None, None]:
        """Raw tokens.
        """
        return utils.find_tokens(self.text,
                                 self.pattern,
                                 self.maximum)

    @property
    def tokens(self) -> Generator[str, None, None]:
        """Processed ngrams.
        """
        tokens = self.unprocessed_tokens
        if self.lowercase:
            tokens = (token.lower() for token in tokens)
        if self.ngrams > 1:
            tokens = utils.get_ngrams(tokens, self.ngrams)
        return tokens

    @property
    def types(self):
        """Document vocabulary.
        """
        return set(self.tokens)

    @property
    def len(self):
        """Token lengths.
        """
        return np.array([len(token) for token in self.tokens])

    @property
    def mean_len(self):
        """Arithmetic mean of token lengths.
        """
        return self.len.mean()

    @property
    def sum_tokens(self):
        """Number of tokens.
        """
        return len(list(self.tokens))

    @property
    def sum_types(self):
        """Number of types.
        """
        return len(self.types)

    @property
    def bow(self):
        """Bag-of-words representation.
        """
        return utils.count_tokens(self.tokens)

    @property
    def rel_freqs(self):
        """Bag-of-words representation with relative frequencies.
        """
        return self.bow / self.sum_tokens

    def mfw(self, n: int = 10, rel_freqs: bool = False, as_list: bool = True) -> Union[List[str], pd.Series]:
        """Most frequent words.

        Parameters:
            n: Number of most frequent words.
            rel_freqs: If True, use relative frequencies for sorting.
            as_list: If True, return as a list, otherwise as pandas Series
                with frequencies.
        """
        freqs = self.bow.sort_values(ascending=False)
        if rel_freqs:
            freqs = freqs.iloc[:n] / self.sum_tokens
        else:
            freqs = freqs.iloc[:n]
        if as_list:
            return list(freqs.index)
        else:
            return freqs

    @property
    def hapax(self):
        """Hapax legomena.
        """
        freqs = self.bow
        return list(freqs[freqs == 1].index)

    def window(self, n: int = 1000) -> Generator[pd.Series, None, None]:
        """Iterate with a sliding window over tokens.

        Parameters:
            n: Window size.
        """
        tokens = list(self.tokens)
        if n > self.sum_tokens:
            n = self.sum_tokens
            logger.warning("{} > number of tokens in document. Setting n = number of tokens.")
        for i in range(int(self.sum_tokens / n)):
            yield utils.count_tokens(tokens[i * n:(i * n) + n])

    @property
    def freq_spectrum(self):
        """Counted occurring frequencies.
        """
        return self.bow.value_counts()

    @staticmethod
    def drop(tokens, features: Iterable[str]) -> Generator[str, None, None]:
        """Drop features from tokens.
        """
        return (token for token in tokens if token not in features)

    def paragraphs(self, sep: Union[re.compile, str] = re.compile(r"\n")) -> Iterable[str]:
        """Paragraphs as separate entities.
        """
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        return filter(None, splitted)

    def segments(self, segment_size: int = 1000, tolerance: float = 0.05,
                     flatten_chunks: bool = True) -> Iterable[List[str]]:
        """Segments as separate entities.
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

    def bootstrap(self, measure: str = "ttr", window: int = 1000, **kwargs) -> Generator[int, None, None]:
        """Iterate with sliding window over tokens and apply a complexity measure.

        Parameters:
            measure: Use `help(cophi_toolbox)` for an extensive description 
                on available complexity measures.
            window: Window size.
            **kwargs: Additional parameter for :func:`complexity.orlov_z`.
        """
        for chunk in self.window(window):
            count = utils._count(chunk, measure)
            if measure == "ttr":
                yield complexity.ttr(**count)
            elif measure == "guiraud_r":
                yield complexity.guiraud_r(**count)
            elif measure == "herdan_c":
                yield complexity.herdan_c(**count)
            elif measure == "dugast_k":
                yield complexity.dugast_k(**count)
            elif measure == "maas_a2":
                yield complexity.maas_a2(**count)
            elif measure == "tuldava_ln":
                yield complexity.tuldava_ln(**count)
            elif measure == "brunet_w":
                yield complexity.brunet_w(**count)
            elif measure == "cttr":
                yield complexity.cttr(**count)
            elif measure == "summer_s":
                yield complexity.summer_s(**count)
            elif measure == "sichel_s":
                yield complexity.sichel_s(**count)
            elif measure == "michea_m":
                yield complexity.michea_m(**count)
            elif measure == "honore_h":
                yield complexity.honore_h(**count)
            elif measure == "entropy":
                yield complexity.entropy(**count)
            elif measure == "yule_k":
                yield complexity.yule_k(**count)
            elif measure == "simpson_d":
                yield complexity.simpson_d(**count)
            elif measure == "herdan_vm":
                yield complexity.herdan_vm(**count)
            elif measure == "orlov_z":
                yield complexity.orlov_z(**count, **kwargs)
            else:
                raise NotImplementedError("The measure '{}' is not implemented.".format(measure))

    def complexity(self, measure: str = "ttr", window: Optional[int]=None, **kwargs):
        """Calculate complexity, optionally with a sliding window.

        Parameters:
            measure: Use `help(cophi_toolbox)` for an extensive description 
                on available complexity measures.
            window: Window size.
            **kwargs: Additional parameter for :func:`complexity.orlov_z`.
        """
        if not window:
            count = utils._count(self.bow, measure)
        if measure == "ttr":
            if window:
                sttr = list(self.bootstrap(measure, window))
                return np.array(sttr).mean(), complexity.ci(sttr)
            else:
                return complexity.ttr(**count)
        elif measure == "guiraud_r":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.guiraud_r(**count)
        elif measure == "herdan_c":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.herdan_c(**count)
        elif measure == "dugast_k":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.dugast_k(**count)
        elif measure == "maas_a2":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.maas_a2(**count)
        elif measure == "tuldava_ln":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.tuldava_ln(**count)
        elif measure == "brunet_w":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.brunet_w(**count)
        elif measure == "cttr":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.cttr(**count)
        elif measure == "summer_s":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.summer_s(**count)
        elif measure == "sichel_s":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.sichel_s(**count)
        elif measure == "michea_m":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.michea_m(**count)
        elif measure == "honore_h":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.honore_h(**count)
        elif measure == "entropy":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.entropy(**count)
        elif measure == "yule_k":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.yule_k(**count)
        elif measure == "simpson_d":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.simpson_d(**count)
        elif measure == "herdan_vm":
            if window:
                return np.array(list(self.bootstrap(measure, window))).mean()
            else:
                return complexity.herdan_vm(**count)
        elif measure == "orlov_z":
            if window:
                return np.array(list(self.bootstrap(measure, window, **kwargs))).mean()
            else:
                return complexity.orlov_z(**count, **kwargs)
        else:
            raise NotImplementedError("The measure '{}' is not implemented.".format(measure))

@dataclass
class Corpus:
    documents: Iterable[Document]
    sparse: bool = False

    def __post_init__(self):
        if self.sparse:
            raise NotImplementedError
        else:
            matrix = pd.DataFrame
        self.dtm = matrix({document.title: document.bow
                           for document in self.documents}).T.fillna(0).astype(int)

    @staticmethod
    def map_metadata(matrix: Union[pd.DataFrame, pd.Series], metadata: pd.DataFrame, uuid: str = "uuid",
                     fields: List[str] = ["title"], sep: str = "_") -> pd.DataFrame:
        """Map metadata with a UUID.

        Parameters:
            matrix: Matrix to map with.
            metadata: Matrix with metadata, one row corresponds to one document.
            uuid: The connecting UUID between `matrix` and `metadata`.
            fields: One or more columns of `metadata`.
            sep: Glue multiple `fields` with this separator together.
        """
        matrix = matrix.copy()  # do not work on original object itself
        document_uuid = metadata[uuid]
        index = metadata[fields[0]].astype(str)
        if len(fields) > 1:
            for field in fields[1:]:
                index = index + sep + metadata[field].astype(str)
        document_uuid.index = index
        matrix.index = document_uuid.to_dict()
        return matrix

    @property
    def size(self):
        """Number of documents and types.
        """
        return pd.Series(self.dtm.shape, index=["documents", "types"])

    @property
    def freq_spectrum(self):
        """Frequency spectrum of types.
        """
        return self.dtm.sum(axis=0).value_counts()

    @property
    def types(self):
        """Corpus vocabulary.
        """
        return list(self.dtm.columns)

    @staticmethod
    def sort(dtm) -> pd.DataFrame:
        """Descending sorted document-term matrix.
        """
        return dtm.iloc[:, (-dtm.sum()).argsort()]

    def mfw(self, n: int = 100, rel_freqs: bool = True, as_list: bool = True) -> Union[List[str], pd.Series]:
        """Most frequent words.

        Parameters:
            n: Number of most frequent words.
            rel_freqs: If True, use relative frequencies for sorting.
            as_list: If True, return as a list, otherwise as pandas Series
                with frequencies.
        """
        dtm = self.sort(self.dtm)
        if rel_freqs:
            mfw = dtm.iloc[:, :n].div(self.dtm.sum(axis=1), axis=0)
        else:
            mfw = dtm.iloc[:, :n]
        if as_list:
            return list(mfw.columns)
        else:
            return mfw.sum()

    @property
    def hapax(self):
        """Hapax legomena.
        """
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
        """Drop features from document-term matrix.
        """
        features = [token for token in features if token in dtm.columns]
        return dtm.drop(features, axis=1)

    @property
    def zscores(self):
        """Standardized document-term matrix.

        Used formula:
        .. math::
            z_x = \frac{x - \mu}{\sigma}
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

        Used formula:
        .. math::
            tf-idf_{t,d} \; = \; tf_{t,d} \times idf_t \; = \; tf_{t,d} \times log(\frac{N}{df_t})
        """
        tf = self.rel_freqs
        idf = self.size["documents"] / self.dtm.astype(bool).sum(axis=0)
        return tf * np.log(idf)

    @property
    def sum_types(self):
        """Summed type frequencies per document.
        """
        return self.dtm.replace(0, np.nan).count(axis=1)

    @property
    def sum_tokens(self):
        """Summed token frequencies per document.
        """
        return self.dtm.sum(axis=1)

    def complexity(self, window: int, measure: str = "ttr"):
        """Calculate complexity for each document with a sliding window.

        Parameters:
            window: Window size.
            measure: Use `help(cophi_toolbox)` for an extensive description 
                on available complexity measures.
            **kwargs: Additional parameter for :func:`complexity.orlov_z`.
        """
        if measure == "ttr":
            c = pd.DataFrame()
        else:
            c = pd.Series()
        for document in self.documents:
            if measure == "ttr":
                sttr, ci = document.complexity(measure, window)
                c = c.append(pd.DataFrame({"sttr": sttr, "ci": ci}, index=[document.title]))
            else:
                c[document.title] = document.complexity(measure, window)
        return c

    @property
    def ttr(self):
        """Type-Token Ratio.
        """
        return complexity.ttr(self.size["types"], self.sum_tokens.sum())

    @property
    def guiraud_r(self) -> float:
        """Guiraud’s index of lexical richness (1954).

        Used formula:
        .. math::
            r = \frac{V}{\sqrt{N}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def herdan_c(self) -> float:
        """Herdan’s index of lexical richness (1960, 1964).

        Used formula:
        .. math::
            c = \frac{\log{V}}{\log{N}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def dugast_k(self) -> float:
        """Dugast’s uber index (1979).

        Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def maas_a2(self) -> float:
        """Maas’ index of lexical richness (1972).
        
        Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def tuldava_ln(self):
        """Tuldava’s index of lexical richness (1977).
        
        Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def brunet_w(self):
        """Get Brunet’s index of lexical richness (1978).
        
        Used formula:
        .. math::
            w = V^{V^{0.172}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def cttr(self):
        """Carroll’s corrected type-token ratio.
        
        Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def summer_s(self):
        """Summer’s index of lexical richness.
        
        Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}
        """
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def sichel_s(self):
        """Sichel’s S (1975)"""
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def michea_m(self):
        """Michéa’s M (1969, 1971)"""
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

    @property
    def honore_h(self):
        """Honoré's H (1979)"""
        return complexity.sichel_s(self.size["types"], self.sum_tokens.sum())

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
        """Yule’s K (1944).
        
        Used formula:
        .. math::
            K = 10^4 \times \frac{(\sum_{X=1}^{X}{{f_X}X^2}) - N}{N^2}
        """
        a = self.freq_spectrum.index / self.sum_tokens.sum()
        b = 1 / self.sum_tokens.sum()
        return 10 ** 4 * ((self.freq_spectrum * a ** 2) - b).sum()

    @property
    def simpson_d(self):
        """Simpson’s D.

        Used formula (where :math:`N` is the number of tokens, and :math:`V` the number of types):
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = self.freq_spectrum.index - 1
        return (self.freq_spectrum * a * (b / (self.sum_tokens.sum() - 1))).sum()

    @property
    def herdan_vm(self):
        """Herdan’s VM (1955).

        Used formula (where :math:`N` is the number of tokens, and :math:`V` the number of types):
        .. math::
            \mbox
        """
        a = self.freq_spectrum / self.sum_tokens.sum()
        b = 1 / self.sum_types.sum().sum()
        return np.sqrt(((self.freq_spectrum * a ** 2) - b).sum())

    def orlov_z(self, max_iterations: int = 100, min_tolerance: int = 1) -> pd.Series:
        """Orlov’s Z.
        """
        return complexity.orlov_z(self.sum_tokens, self.size["types"], self.freq_spectrum, max_iterations, min_tolerance)
