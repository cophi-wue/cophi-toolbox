"""
cophi_toolbox
~~~~~~~~~~~~~

`cophi_toolbox` is an NLP preprocessing library for handling and processing text data
on three different levels.

On the document level:

    >>> import cophi_toolbox
    >>> d = cophi_toolbox.document("corpus/document.txt")
    >>> d.name
    "document"
    >>> d.document
    "Everything's gone green."

or, on the token level:

    >>> t = cophi_toolbox.token("Everything's gone green.")
    >>> t.document
    "Everything's gone green."
    >>> list(t.tokens)
    ["everything's", 'gone', 'green']

or, on the corpus level:

    >>> import pandas as pd
    >>> s = pd.Series(["everything's", "gone", "green", "green"], name="doc")
    >>> c = corpus([s])
    >>> c.model
         everything's  gone  green
    doc             1     1      2
    >>> c.mfw(threshold=1)  # defaults to 100
    >>> list(c.mfw)  # most frequent words
    ["green"]
    >>> list(c.hl)  # hapax legomena
    ["everything's", "gone"]
"""

from .api import document, token, corpus, pipe
from .model import Document, Token, Corpus
from .utils import find_tokens, count_tokens, get_ngrams, segment_fuzzy
