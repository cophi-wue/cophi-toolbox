"""
cophi.api
~~~~~~~~~

This module implements the high-level API.
"""

import logging
import pathlib
import uuid
import pandas as pd
import cophi.model


logger = logging.getLogger(__name__)


def document(filepath, **kwargs):
    """Read a text file and create a Document object.

    Parameter:
        filepath (str): Path to the text file.
        title (str): Describing title for the document. (optional).
        lowercase (bool): If True, writes all letters in lowercase (optional).
        n (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Returns:
        A Document object.
    """
    textfile = cophi.model.Textfile(filepath)
    return cophi.model.Document(textfile.content, **kwargs)


def corpus(directory, filepath_pattern="*", treat_as=None, encoding="utf-8",
           lowercase=True, n=None, token_pattern=r"\p{L}+\p{P}?\p{L}+",
           maximum=None):
    """Pipe a collection of text files and create a Corpus object.

    Parameters:
        directory (str): Path to the corpus directory.
        filepath_pattern (str): Glob pattern for text files (optional).
        treat_as (str): Treat text files like .txt or .xml (optional).
        encoding (str): Encoding to use for UTF when reading (optional).
        lowercase (bool): If True, writes all letters in lowercase (optional).
        n (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Returns:
        A Corpus model object and a Metadata object.
    """
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    filepaths = directory.rglob(filepath_pattern)

    def lazy_reading(filepaths):
        for filepath in filepaths:
            if filepath.is_file():
                yield cophi.model.Textfile(filepath, treat_as, encoding)

    metadata = cophi.model.Metadata()
    documents = pd.Series()
    for textfile in lazy_reading(filepaths):
        logger.info("Processing '{}' ...".format(textfile.title))
        document_uuid = str(uuid.uuid1())
        text = textfile.content
        document = cophi.model.Document(text,
                                        document_uuid,
                                        token_pattern,
                                        lowercase,
                                        n,
                                        maximum)
        documents[document_uuid] = document
        metadata = metadata.append({"uuid": document_uuid,
                                    "filepath": textfile.filepath,
                                    "parent": textfile.parent,
                                    "title": textfile.title,
                                    "suffix": textfile.filepath.suffix},
                                    ignore_index=True)
    logger.info("Constructing Corpus object ...")
    return cophi.model.Corpus(documents), metadata
