"""
cophi_toolbox
~~~~~~~~~~~~~

`cophi_toolbox` is an NLP preprocessing library for handling and processing text data.

For high-level use, you can pipe files in a directory:

    >>> import cophi_toolbox as ct
    >>> c = ct.pipe("/home/user/corpus")
    >>> c.dtm()  # creating a document-term matrix
    >>> c.model
         everything's  gone  green
    doc             1     1      2

Going lower-level, you can handle text files:

    >>> import cophi_toolbox
    >>> d = cophi_toolbox.document("corpus/document.txt")
    >>> d.name
    "document"
    >>> d.document
    "Everything's gone green."

or, tokens of a document:

    >>> t = cophi_toolbox.token("Everything's gone green.")
    >>> t.document
    "Everything's gone green."
    >>> list(t.tokens)
    ["everything's", 'gone', 'green']

or, the corpus model:

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

You can even use lower low-level helpe functions, e.g.:

    >>> find_tokens("Everything's gone green.")
    <generator object find_tokens at ...>
    >>> get_ngrams(["everything's", "gone", "green", "green"])
    <generator object get_ngrams.<locals>.<genexpr> at ...>
"""

from .api import document, token, corpus, pipe
from .model import Document, Token, Corpus
from .utils import find_tokens, count_tokens, get_ngrams, segment_fuzzy
