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
from typing import Optional, Iterable, Union
from dataclasses import dataclass

from lxml import etree
import pandas as pd
import regex as re

logger = logging.getLogger("cophi_toolbox.model")


@dataclass
class Textfile:
    filepath: str
    treat_as: str = ".txt"
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.treat_as not in {".txt", ".xml"}:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(self.treat_as))

    def __enter__(self):
        self.from_disk()
        return self.text

    def __exit__(self):
        del self.text
        del self.name

    def _read_txt(self):
        """Read plain text file, construct text and name object. Wrapped in :func:`from_disk()`,
        and :func:`_read_xml()`.
        """
        p = pathlib.Path(self.filepath)
        self.name = p.stem
        self.text = p.read_text(encoding=self.encoding)

    def _read_xml(self, parser: etree.XMLParser = etree.XMLParser(), _path: Optional[str] = None,
                 namespaces: dict = dict(tei="http://www.tei-c.org/ns/1.0")):
        """Read and parse XML file, construct text and name object. Wrapped in :func:`from_disk()`.

        Parameters:
            parser: Overwrite default parser with this.
            _path: Evaluate this XPath expression using the text as context node.
            namespaces: Namespaces for the XPath expression.
        """
        self.read_txt()
        tree = etree.fromstring(self.text, parser=parser)
        if _path is None:
            self.text = " ".join(tree.itertext())
        else:
            self.text = " ".join(tree.xpath(_path, namespaces=namespaces))

    def from_disk(self, **kwargs: str):
        """Read text object from a text file.

        Parameters:
            parser: Overwrite default XML parser with this. Only if `treat_as` is `.xml`.
            _path: Evaluate this XPath expression using. Only if `treat_as` is `.xml`.
            namespaces: Namespaces for the XPath expression. Only if `treat_as` is `.xml`.
        """
        if self.treat_as == ".txt":
            self._read_txt()
        elif self.treat_as == ".xml":
            self._read_xml(**kwargs)
        if not self.text:
            logger.warning("Your text object is empty.")

    def to_disk(self, filepath: str):
        """Write text object to a text file.

        Parameters:
            filepath: Path to text file.
        """
        with open(filepath, "w", encoding=self.encoding) as file:
            file.write(self.text)


@dataclass
class Document:
    text: Optional[str] = None
    lowercase: bool = True
    ngrams: int = 1
    pattern: str = r"\p{L}+\p{P}?\p{L}+"
    maximum: Optional[int] = None

    def _tokenize(self):
        """Tokenize text object.
        """
        self.tokens = utils.find_tokens(self.text,
                                        self.pattern,
                                        self.maximum)

    def _postprocess(self):
        """Postprocess tokens object.
        """
        if self.lowercase:
            self.tokens = (token.lower() for token in self.tokens)
        if self.ngrams > 1:
            self.tokens = utils.get_ngrams(self.tokens, self.ngrams)

    def drop(self, features: Iterable[str]):
        """Drop features (tokens, or words) from tokens object.

        Parameters:
            features: Tokens to remove from the tokens object.

        Returns:
            An iterable of tokens.
        """
        tokens = self.tokens
        return (token for token in tokens if token not in features)

    def get_paragraphs(self, sep: Union[re.compile, str] = re.compile(r"\n")):
        """Split text object by paragraphs.

        Parameters:
            sep: Separator between paragraphs.

        Returns:
            An iterable of paragraphs.
        """
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        return filter(None, splitted)

    def get_segments(self, segment_size: int = 1000, tolerance: float = 0.05,
                     flatten_chunks: bool = True):
        """Segment paragraphs of text object, respecting a tolerance threshold value.

        Parameters:
            segment_size: The target size of each segment, in tokens.
            tolerance: How much may the actual segment size differ from
                the segment_size? If ``0 < tolerance < 1``, this is interpreted as a
                fraction of the segment_size, otherwise it is interpreted as an
                absolute number. If ``tolerance < 0``, chunks are never split apart.
            flatten_chunks: If True, undo the effect of the chunker by
                chaining the chunks in each segment, thus each segment consists of
                tokens. This can also be a one-argument function in order to
                customize the un-chunking.

        Returns:
            An iterable of segments.
        """
        segments = utils.segment_fuzzy([self.get_paragraphs()],
                                       segment_size,
                                       tolerance)
        if flatten_chunks:
            if not callable(flatten_chunks):
                def flatten_chunks(segment):
                    return list(itertools.chain.from_iterable(segment))
            segments = map(flatten_chunks, segments)
        return segments

    def to_disk(self, filepath: str, encoding: str = "utf-8", sep: str = "\n"):
        """Write tokens object to a text file, tokens separated with `sep`.

        Parameters:
            filepath: Path to text file.
            encoding: Encoding to use for UTF when writing.
            sep: Separator between two tokens.
        """
        with open(filepath, "w", encoding=encoding) as file:
            for token in self.tokens:
                file.write(token + sep)

    def from_disk(self, filepath: str, encoding: str, sep: str = "\n"):
        """Read tokens object from a text file, tokens separated with `sep`.

        Parameters:
            filepath: Path to text file.
            encoding: Encoding to use for UTF when reading.
            sep: Separator between two tokens.
        """
        with open(filepath, "r", encoding=encoding) as file:
            self.tokens = filter(None, file.read().split(sep))


@dataclass
class Corpus:
    tokens: Optional[Iterable[pd.Series]] = None

    def __post_init__(self):
        self.size = len(self.tokens)

    def dtm(self):
        """Create classic document-term matrix, construct model object.
        """
        self.model = pd.SparseDataFrame({document.name: utils.count_tokens(document)
                                         for document in self.tokens}).T.fillna(0)

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
        """
        return list(self.model.loc[:, self.model.max() == 1].columns)

    def drop(self, features: Iterable[str]):
        """Drop features (tokens, or words) from model object.

        Parameters:
            features: Tokens to remove from the tokens object.
        """
        features = [token for token in features if token in self.model.columns]
        self.model = self.model.drop(features, axis=1)

    def get_zscores(self):
        """Calculate z-scores for word frequencies.

        Returns:
            A pandas DataFrame document-term matrix.
        """
        return (self.model - self.model.mean()) / self.model.std()

    def get_rel_freqs(self):
        """Calculate relative word frequencies.

        Returns:
            A pandas DataFrame document-term matrix.
        """
        return self.model.div(self.model.sum(axis=1).to_sparse(), axis=0)

    def get_tfidf(self):
        """Calculate TF-IDF.

        Used formula is

        .. math::
            \sum_{i=1}^{\\infty} x_{i}

        Returns:
            A pandas DataFrame document-term matrix.
        """
        tf = self.get_rel()
        idf = math.log(self.size / self.model.astype(bool).sum(axis=0))
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