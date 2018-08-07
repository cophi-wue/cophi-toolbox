"""
cophi_toolbox
~~~~~~~~~~~~~

`cophi_toolbox` is an NLP preprocessing library for handling and processing text data.

"""

from .api import pipe
from .model import Textfile, Document, Corpus
from .utils import find_tokens, count_tokens, construct_ngrams, segment_fuzzy