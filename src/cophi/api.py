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
    """Read text file and create a Document object.

    Parameter:
        filepath (str): Path to the text file.
        title (str): Text file’s title (optional).
        lowercase (bool): If True, lowercase all letters (optional).
        ngrams (int): Number of tokens per ngram (optional).
        pattern (str): Regex pattern for one token (optional).
        maximum (int): If not None, stop reading after that much tokens (optional).

    Returns:
        A Document object.
    """
    textfile = model.Textfile(filepath)
    return cophi.model.Document(textfile.content, **kwargs)

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
        document = model.Document(text, document_id, lowercase, ngrams, token_pattern, maximum)
        documents[document_id] = document
        metadata = metadata.append({"uuid": document_id,
                                    "filepath": textfile.filepath,
                                    "parent": textfile.parent,
                                    "title": textfile.title,
                                    "suffix": textfile.filepath.suffix},
                                    ignore_index=True)
    return model.Corpus(documents), metadata
