"""
cophi.api
~~~~~~~~~

This module implements the high-level API.
"""

import uuid
import pathlib

import pandas as pd

import cophi.model


def document(filepath, **kwargs):
    """Read a text file and create a Document object.

    Parameter:
        filepath (str): Path to the text file.
        title (str): Text file’s title (optional).
        lowercase (bool): If True, all letters are lowercase (optional).
        ngrams (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Returns:
        A Document object.
    """
    textfile = model.Textfile(filepath)
    return cophi.model.Document(textfile.content, **kwargs)


def corpus(directory, filepath_pattern="*.*", treat_as=None, encoding="utf-8",
           lowercase=True, ngrams=1, token_pattern=r"\p{L}+\p{P}?\p{L}+",
           maximum=None):
    """Pipe a collection of text files and create a Corpus object.

    Parameters:
        directory (str): Path to the corpus directory.
        filepath_pattern (str): Glob pattern for text files (optional).
        treat_as (str): Treat text files like this suffix (optional).
        encoding (str): Encoding to use for UTF when reading (optional).
        lowercase (bool): If True, all letters are lowercase (optional).
        ngrams (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Returns:
        A Corpus model object and a Metadata object.
    """
    if isinstance(directory, str):
        directory = pathlib.Path(directory)
    filepaths = directory.glob(filepath_pattern)

    def lazy_reading(filepaths):
        for filepath in filepaths:
            yield cophi.model.Textfile(filepath, treat_as, encoding)

    metadata = cophi.model.Metadata()
    documents = pd.Series()
    for textfile in lazy_reading(filepaths):
        document_id = str(uuid.uuid1())
        text = textfile.content
        document = cophi.model.Document(text,
                                        document_id,
                                        lowercase,
                                        ngrams,
                                        token_pattern,
                                        maximum)
        documents[document_id] = document
        metadata = metadata.append({"uuid": document_id,
                                    "filepath": textfile.filepath,
                                    "parent": textfile.parent,
                                    "title": textfile.title,
                                    "suffix": textfile.filepath.suffix},
                                    ignore_index=True)
    return cophi.model.Corpus(documents), metadata
