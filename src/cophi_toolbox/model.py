"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides corpus model objects to manage and process 
text data in Python.
"""

from . import utils

import pathlib
import collections
import itertools
from typing import Optional, Iterable
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
        self.tokens = utils.find_tokens(self.document,
                                        self.pattern,
                                        self.maximum)

    def postprocess(self):
        if self.lowercase:
            self.tokens = (token.lower() for token in self.tokens)
        if self.ngrams > 1:
            self.tokens = utils.get_ngrams(self.tokens, n=self.ngrams)

    def drop(self, features: Iterable[str]):
        self.tokens = (token for token in self.tokens if token not in features)

    def to_disk(self, filepath: str, encoding: str, sep: str = "\n"):
        with pathlib.Path(filepath).open("w", encoding=encoding) as file:
            for token in self.tokens:
                file.write(token + sep)

    def from_disk(self, filepath: str, encoding: str, sep: str = "\n"):
        with pathlib.Path(filepath).open("r", encoding=encoding) as file:
            self.tokens = iter(file.read().split(sep))


@dataclass
class Document:
    filepath: str
    treat_as: str = ".txt"
    encoding: str = "utf-8"

    def read_txt(self):
        p = pathlib.Path(self.filepath)
        self.name = p.stem
        self.document = p.read_text(encoding=self.encoding)

    def read_xml(self, parser: etree.Parser = etree.XMLParser(), _path: str = None,
                 namespaces: dict = dict(tei="http://www.tei-c.org/ns/1.0")):
        self.read_txt(self.filepath)
        tree = etree.fromstring(self.document, parser=parser)
        if _path is None:
            self.document = " ".join(tree.itertext())
        else:
            self.document = tree.xpath(_path, namespaces=namespaces)

    def from_disk(self):
        if self.treat_as == ".txt":
            self.read_txt()
        elif self.treat_as == ".xml":
            self.read_xml()
        else:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(self.treat_as))

    def to_disk(self, filepath: str):
        with pathlib.Path(filepath).open("w", encoding=encoding) as file:
            file.write(self.document)

    def paragraphs(self, sep: re.compile = re.compile(r"\n")):
        if not hasattr(sep, 'match'):
            sep = re.compile(sep)
        splitted = sep.split(self.document)
        self.paragraphs = list(filter(str.strip, splitted))

    def segment(self, segment_size: int = 1000, tolerance: float = 0.05,
                flatten_chunks: bool = True):
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
    tokens: Iterable[pd.Series[str]]

    def dtm(self):
        self.model = pd.DataFrame({document.name: utils.count_tokens(document)
                                   for document in self.tokens}).T.fillna(0)

    def mm(self):
        self.model = None

    def sort(self, ascending: bool = False):
        self.model = self.model.loc[:, self.model.sum().sort_values(ascending=ascending).index]

    def get_mfw(self, threshold: int = 100):
        self.mfw = list(self.model.iloc[:, :threshold].columns)

    def get_hl(self):
        self.hl = list(self.model.loc[:, self.model.max() == 1].columns)

    def drop(self, features: Iterable[str]):
        features = [token for token in features if token in self.model.columns]
        self.model = self.model.drop(features, axis=1)

    def to_disk(self, filepath: str, **kwargs: str):
        self.model.to_csv(filepath, **kwargs)

    def from_disk(self, filepath: str, **kwargs: str):
        self.model = pd.read_csv(filepath, **kwargs)