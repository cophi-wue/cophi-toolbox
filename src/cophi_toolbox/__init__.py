"""
cophi_toolbox
~~~~~~~~~~~~~

This is an NLP preprocessing library for handling, processing and modeling text data.
"""

from .api import pipe
from .model import Textfile, Document, Corpus
from .utils import find_tokens, count_tokens, construct_ngrams, segment_fuzzy