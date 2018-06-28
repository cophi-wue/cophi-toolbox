"""
cophi_toolbox.model
~~~~~~~~~~~~~~~~~~~

This module provides corpus model objects to manage and process 
text data in Python.
"""

import logging
import pathlib
import utils
import collections
from typing import Generator, Optional, List, Iterable, Union
from dataclasses import dataclass
from lxml import etree
import pandas as pd
import regex as re
import itertools


@dataclass
class Token:
    document: str
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


@dataclass
class Document:
    filepath: str
    treat_as: str = ".txt"
    encoding: str = "utf-8"

    def txt(self) -> str:
        p = pathlib.Path(self.filepath)
        self.name = p.stem
        self.text = p.read_text(encoding=self.encoding)

    def xml(self, parser=etree.XMLParser, _path=None,
                 namespaces=dict(tei="http://www.tei-c.org/ns/1.0")) -> Union[list, str]:
        self.read_txt(self.filepath)
        tree = etree.fromstring(self.text, parser=parser)
        if _path is None:
            self.text = " ".join(tree.itertext())
        else:
            self.text = tree.xpath(_path, namespaces=namespaces)

    def read(self) -> pd.Series:
        if self.treat_as == ".txt":
            self.txt()
        elif self.treat_as == ".xml":
            self.xml()
        else:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(self.treat_as))

    def split_paragraphs(self, sep=re.compile(r"\n")):
        if not hasattr(sep, 'match'):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        self.paragraphs = list(filter(str.strip, splitted))

    def segment(self, segment_size=1000, tolerance=0.05, flatten_chunks=True):
        segments = utils.segment_fuzzy([self.paragraphs],
                                       segment_size,
                                       tolerance)
        if flatten_chunks:
            if not callable(flatten_chunks):
                def flatten_chunks(segment):
                    return list(itertools.chain.from_iterable(segment))
            segments = map(flatten_chunks, segments)
        self.segments = list(segments)


@dataclass
class Corpus:
    tokenized_documents: Iterable[Iterable[str]]

    def document_term_matrix(self) -> pd.DataFrame:
        self.model = pd.DataFrame({document.name: utils.count_tokens(document)
                                   for document in self.tokenized_documents}).T
        self.model = self.model.fillna(0)

    def get_mfw(self):
        self.mfw = None

    def get_hapax_legomena(self):
        self.hl = None

    def drop_features(self, features: Iterable[str]):
        self.dtm = None