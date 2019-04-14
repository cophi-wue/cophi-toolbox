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
           maximum=None, metadata=False):
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
        metadata (bool): Extract metadata from filenames (optional).

    Returns:
        A Corpus model object and optionally a Metadata object.
    """
    if not isinstance(directory, pathlib.Path):
        directory = pathlib.Path(directory)
    filepaths = directory.rglob(filepath_pattern)

    def lazy_reading(filepaths):
        for filepath in filepaths:
            if filepath.is_file() and ".git" not in str(filepath):
                yield cophi.model.Textfile(filepath, treat_as, encoding)

    if metadata:
        metadata_ = cophi.model.Metadata()
    documents = pd.Series()
    for textfile in lazy_reading(filepaths):
        logger.info("Processing '{}' ...".format(textfile.title))
        title = str(uuid.uuid1()) if metadata else textfile.title
        text = textfile.content
        document = cophi.model.Document(text,
                                        title,
                                        token_pattern,
                                        lowercase,
                                        n,
                                        maximum)
        documents[title] = document
        if metadata:
            metadata_ = metadata_.append({"uuid": title,
                                          "filepath": textfile.filepath,
                                          "parent": textfile.parent,
                                          "title": textfile.title,
                                          "suffix": textfile.filepath.suffix},
                                          ignore_index=True)
    logger.info("Constructing Corpus object ...")
    if metadata:
        return cophi.model.Corpus(documents), metadata
    else:
        return cophi.model.Corpus(documents)

def export(dtm, filepath, format="text"):
    """Export a document-term matrix.

    Parameters:
        dtm: A document-term matrix.
        filepath: Path to output file. Possibel values are `plaintext`/`text` or
            `svmlight`.
        format: File format.
    """
    if format.lower() in {"plaintext", "text"}:
        cophi.model.Corpus.plaintext(dtm, filepath)
    elif format.lower() in {"svmlight"}:
        cophi.model.Corpus.svmlight(dtm, filepath)
    else:
        raise ValueError("'{}' is no supported file format.".format(format))
