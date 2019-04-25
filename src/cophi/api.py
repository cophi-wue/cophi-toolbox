"""
cophi.api
~~~~~~~~~

This module implements the high-level API.
"""

import logging
from pathlib import Path
import uuid

import pandas as pd

from cophi import dkpro, text


logger = logging.getLogger(__name__)


def document(filepath, lemma=False, pos=None, jar="ddw-0.4.6.jar",
             language="de", **kwargs):
    """Read a text file and create a Document object.

    Parameter:
        filepath (str): Path to the text file.
        lemma (bool): If True, lemmatize text (optional).
        pos (list): If not None, filter POS tags (optional).
        jar (str): Path to DARIAH-DKPro-Wrapper JAR file (optional).
        language (str): Language of text (optional).
        title (str): Describing title for the document (optional).
        lowercase (bool): If True, writes all letters in lowercase (optional).
        n (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Returns:
        A Document object.
    """
    if lemma or pos:
        return dkpro.pipe(filepath, jar, language, lemma, pos, **kwargs)
    else:
        filepath = text.model.Textfile(filepath)
        return text.model.Document(filepath.content, **kwargs)


def corpus(directory, filepath_pattern="*.txt", treat_as=None, encoding="utf-8",
           lowercase=True, n=None, token_pattern=r"\p{L}+\p{P}?\p{L}+",
           maximum=None, metadata=True, lemma=False, pos=None,
           jar="ddw-0.4.6.jar", language="de"):
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
        lemma (bool): If True, lemmatize text (optional).
        pos (list): If not None, filter POS tags (optional).
        jar (str): Path to DARIAH-DKPro-Wrapper JAR file (optional).
        language (str): Language of text (optional).

    Returns:
        A Corpus model object and optionally a Metadata object.
    """
    filepaths = Path(directory).rglob(filepath_pattern)

    def lazy_processing(filepaths, **kwargs):
        for filepath in filepaths:
            logger.info("Processing '{}' ...".format(filepath.stem))
            if filepath.is_file():
                if lemma or pos:
                    document = dkpro.pipe(filepath,
                                          jar,
                                          language,
                                          lemma,
                                          pos,
                                          **kwargs)
                else:
                    textfile = text.model.Textfile(filepath, treat_as, encoding)
                    document = text.model.Document(textfile.content,
                                                   textfile.title,
                                                   **kwargs)
                yield filepath, document

    if metadata:
        metadata_ = text.model.Metadata()

    documents = pd.Series()
    for filepath, document in lazy_processing(filepaths,
                                              token_pattern=token_pattern,
                                              lowercase=lowercase,
                                              n=n,
                                              maximum=maximum):
        title = document.title
        if metadata:
            title = str(uuid.uuid1())
            document.title = title
            metadata_ = metadata_.append({"uuid": title,
                                          "filepath": str(filepath),
                                          "parent": str(filepath.parent),
                                          "title": filepath.stem,
                                          "suffix": filepath.suffix},
                                          ignore_index=True)
        documents[title] = document
    logger.info("Constructing Corpus object ...")
    if metadata:
        return text.model.Corpus(documents), metadata_
    else:
        return text.model.Corpus(documents)
