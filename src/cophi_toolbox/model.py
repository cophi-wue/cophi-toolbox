"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides low-level corpus model objects to manage and 
process text data in Python.
"""

from . import utils

import pathlib
import collections
import itertools
from typing import Optional, Iterable, Union
from dataclasses import dataclass

from lxml import etree
import pandas as pd
import regex as re


@dataclass
class Token:
    document: Optional[str] = None
    pattern: str = r"\p{L}+\p{P}?\p{L}+"
    maximum: Optional[int] = None
    lowercase: bool = True
    ngrams: int = 1

    def tokenize(self):
        """Tokenize document object.
        """
        self.tokens = utils.find_tokens(self.document,
                                        self.pattern,
                                        self.maximum)

    def postprocess(self):
        """Postprocess tokens object.
        """
        if self.lowercase:
            self.tokens = (token.lower() for token in self.tokens)
        if self.ngrams > 1:
            self.tokens = utils.get_ngrams(self.tokens, n=self.ngrams)

    def drop(self, features: Iterable[str]):
        """Drop features (tokens, or words) from tokens object.

        Parameters:
            features: Tokens to remove from the tokens object.
        """
        self.tokens = (token for token in self.tokens if token not in features)

    def to_disk(self, filepath: str, encoding: str, sep: str = "\n"):
        """Write tokens object to a text file, tokens separated with `sep`.

        Parameters:
            filepath: Path to text file.
            encoding: Encoding to use for UTF when writing.
            sep: Separator between two tokens.
        """
        with pathlib.Path(filepath).open("w", encoding=encoding) as file:
            for token in self.tokens:
                file.write(token + sep)

    def from_disk(self, filepath: str, encoding: str, sep: str = "\n"):
        """Read tokens object from a text file, tokens separated with `sep`.

        Parameters:
            filepath: Path to text file.
            encoding: Encoding to use for UTF when reading.
            sep: Separator between two tokens.
        """
        with pathlib.Path(filepath).open("r", encoding=encoding) as file:
            self.tokens = iter(file.read().split(sep))


@dataclass
class Document:
    filepath: Optional[str] = None
    treat_as: str = ".txt"
    encoding: str = "utf-8"

    def __post_init__(self):
        if self.treat_as not in [".txt", ".xml"]:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(self.treat_as))

    def read_txt(self):
        """Read plain text file from disk, construct document object, wrapped in :func:`from_disk()`.
        """
        p = pathlib.Path(self.filepath)
        self.name = p.stem
        self.document = p.read_text(encoding=self.encoding)

    def read_xml(self, parser: etree.XMLParser = etree.XMLParser(), _path: Optional[str] = None,
                 namespaces: dict = dict(tei="http://www.tei-c.org/ns/1.0")):
        """Read and parse XML file from disk, construct document object, wrapped in :func:`from_disk()`.

        Parameters:
            parser: Overwrite default parser with this.
            _path: Evaluate this XPath expression using the document as context node.
            namespaces: Namespaces for the XPath expression.
        """
        self.read_txt(self.filepath)
        tree = etree.fromstring(self.document, parser=parser)
        if _path is None:
            self.document = " ".join(tree.itertext())
        else:
            self.document = tree.xpath(_path, namespaces=namespaces)

    def from_disk(self):
        """Read document object from a text file.
        """
        if self.treat_as == ".txt":
            self.read_txt()
        elif self.treat_as == ".xml":
            self.read_xml()

    def to_disk(self, filepath: str):
        """Write document object to a text file.

        Parameters:
            filepath: Path to text file.
        """
        with pathlib.Path(filepath).open("w", encoding=encoding) as file:
            file.write(self.document)

    def get_paragraphs(self, sep: Union[re.compile, str] = re.compile(r"\n")):
        """Split document object by paragraphs, construct chunks object.

        Parameters:
            sep: Separator between paragraphs.
        """
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.document)
        self.chunks = filter(str.strip, splitted)

    def get_segments(self, segment_size: int = 1000, tolerance: float = 0.05,
                flatten_chunks: bool = True):
        """Segment chunks object, respecting a tolerance threshold value.

        Parameters:
            segment_size: 
            tolerance: 
            flatten_chunks: 
        """
        segments = utils.segment_fuzzy([self.paragraphs],
                                       segment_size,
                                       tolerance)
        if flatten_chunks:
            if not callable(flatten_chunks):
                def flatten_chunks(segment):
                    return list(itertools.chain.from_iterable(segment))
            segments = map(flatten_chunks, segments)
        self.segments = segments


@dataclass
class Corpus:
    tokens: Optional[Iterable[pd.Series]] = None

    def dtm(self):
        """Create classic document-term matrix, construct model object.

        Note:
            * Not recommended for very extensive corpora.
        """
        self.model = pd.DataFrame({document.name: utils.count_tokens(document)
                                   for document in self.tokens}).T.fillna(0)

    def mm(self):
        """Create Matrix Market corpus model, construct model object.

        Note:
            * Recommended for very extensive corpora.
        """
        raise NotImplementedError

    def sort(self, ascending: bool = False):
        """Sort corpus model by frequency.
        """
        self.model = self.model.loc[:, self.model.sum().sort_values(ascending=ascending).index]

    def get_mfw(self, threshold: int = 100):
        """Get the most frequent words from corpus object.
        """
        self.sort()
        self.mfw = iter(self.model.iloc[:, :threshold].columns)

    def get_hl(self):
        """Get hapax legomena from corpus object.
        """
        self.hl = iter(self.model.loc[:, self.model.max() == 1].columns)

    def drop(self, features: Iterable[str]):
        """Drop features (tokens, or words) from model object.

        Parameters:
            features: Tokens to remove from the tokens object.
        """
        features = [token for token in features if token in self.model.columns]
        self.model = self.model.drop(features, axis=1)

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
        self.model = pd.read_csv(filepath, **kwargs)