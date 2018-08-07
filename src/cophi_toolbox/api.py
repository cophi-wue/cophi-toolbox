"""
cophi_toolbox.api
~~~~~~~~~~~~~~~~~

This module implements the high-level API.
"""

from . import model

import uuid
import pathlib
from typing import Optional, Union, Tuple

import pandas as pd


def pipe(directory: Union[str, pathlib.Path], pathname_pattern: str = "*.*",
         treat_as: Optional[str] = None, encoding: str = "utf-8", lowercase: bool = True,
         ngrams: int = 1, token_pattern: str = r"\p{L}+\p{P}?\p{L}+",
         maximum: Optional[int] = None) -> Tuple[model.Corpus, pd.DataFrame]:
    """Pipe a collection of text files through multiple NLP tasks.

    Parameters:
        directory: Path to the corpus directory.
        pathname_pattern: Glob pattern for text files.
        treat_as: Treat text files like this suffix. If None, `pathname_pattern` is considered.
        encoding: Encoding to use for UTF when reading.
        lowercase: If True, all letters are lowercase.
        ngrams: The `n` in ngram, e.g. 1 for unigram, 2 for bigram, etc.
        token_pattern: Regex pattern for a token.
        maximum: If not None, stop tokenizing after that much tokens.

    Returns:
        A Corpus model object and a pandas DataFrame with metadata.
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    filepaths = directory.glob(pathname_pattern)

    def lazy_reading(filepaths):
        for filepath in filepaths:
            yield model.Textfile(filepath, treat_as, encoding)

    metadata = pd.DataFrame()
    documents = pd.Series()
    for textfile in lazy_reading(filepaths):
        document_id = str(uuid.uuid1())
        text = textfile.content
        d = model.Document(text, lowercase, ngrams, token_pattern, maximum)
        tokens = pd.Series(d.tokens)
        tokens.name = document_id
        documents[document_id] = tokens
        metadata = metadata.append({"uuid": document_id,
                                    "filepath": textfile.filepath,
                                    "parent": textfile.parent,
                                    "title": textfile.title,
                                    "suffix": textfile.filepath.suffix},
                                    ignore_index=True)
    return model.Corpus(documents), metadata
