"""
cophi_toolbox
~~~~~~~~~~~~~

`cophi_toolbox` is an NLP preprocessing library for handling and processing text data.

For high-level use, you can pipe files in a directory:

    >>> import cophi_toolbox as ct
    >>> c = ct.pipe(directory="/home/user/corpus")
    >>> c.model
         everything's  gone  green
    doc             1     1      2

Going lower-level, you can handle single text files (and easily parse XML):

    >>> d = ct.document(filepath="corpus/document.xml")
    >>> d.name
    "document"
    >>> d.text
    "Everything's gone green."  # used to be XML

or, tokens of a document:

    >>> t = ct.token("Everything's gone green.")
    >>> t.text
    "Everything's gone green."
    >>> list(t.tokens)
    ["everything's", 'gone', 'green']

or, a corpus model:

    >>> import pandas as pd
    >>> s = pd.Series(["everything's", "gone", "green", "green"], name="document")
    >>> c = corpus([s])
    >>> c.model
              everything's  gone  green
    document             1     1      2
    >>> c.mfw(threshold=1)  # defaults to 100
    >>> c.mfw  # show most frequent word
    ["green"]
    >>> c.hl  # show hapax legomena (words occuring only once)
    ["everything's", "gone"]

You can even use lower low-level helper functions, e.g.:

    >>> ct.find_tokens("Everything's gone green.")
    <generator object find_tokens at ...>
    >>> ct.get_ngrams(["everything's", "gone", "green", "green"])
    <generator object get_ngrams.<locals>.<genexpr> at ...>
"""

from .api import textfile, document, corpus, pipe
from .model import Textfile, Document, Corpus
from .utils import find_tokens, count_tokens, get_ngrams, segment_fuzzy
