#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocessing Text Data, Creating Matrices and Cleaning Corpora
***************************************************************

Functions of this module are for **preprocessing purpose**. You can read text \
files, `tokenize <https://en.wikipedia.org/wiki/Tokenization_(lexical_analysis)>`_ \
and segment documents (if a document is chunked into smaller segments, each segment \
counts as one document), create and read `document-term matrices <https://en.wikipedia.org/wiki/Document-term_matrix>`_, \
determine and remove features. Recurrent variable names are based on the following \
conventions:

    * ``corpus`` means an iterable containing at least one ``document``.
    * ``document`` means one single string containing all characters of a text \
    file, including whitespaces, punctuations, numbers, etc.
    * ``dkpro_document`` means a pandas DataFrame containing tokens and additional \
    information, e.g. *part-of-speech tags* or *lemmas*, produced by `DARIAH-DKPro-Wrapper <https://github.com/DARIAH-DE/DARIAH-DKPro-Wrapper>`_.
    * ``tokenized_corpus`` means an iterable containing at least one ``tokenized_document`` \
    or ``dkpro_document``.
    * ``tokenized_document`` means an iterable containing tokens of a ``document``.
    * ``document_labels`` means an iterable containing names of each ``document`` \
    and must have as much elements as ``corpus`` or ``tokenized_corpus`` does.
    * ``document_term_matrix`` means either a pandas DataFrame with rows corresponding to \
    ``document_labels`` and columns to types (distinct tokens in the corpus), whose \
    values are token frequencies, or a pandas DataFrame with a MultiIndex \
    and only one column corresponding to word frequencies. The first column of the \
    MultiIndex corresponds to a document ID (based on ``document_labels``) and the \
    second column to a type ID. The first variant is designed for small and the \
    second for large corpora.
    * ``token2id`` means a dictionary containing a token as key and an unique identifier \
    as key, e.g. ``{'first_document': 0, 'second_document': 1}``.

Contents
********
    * :func:`create_document_term_matrix()` creates a document-term matrix, for either \
    small or large corpora.
    * :func:`filter_pos_tags()` filters a ``dkpro_document`` by specific \
    *part-of-speech tags* and returns either tokens or, if available, lemmas.
    * :func:`find_hapax_legomena()` determines *hapax legomena* based on frequencies \
    of a ``document_term_matrix``.
    * :func:`list_mfw()` determines *most frequent words* based on frequencies \
    of a ``document_term_matrix``.
    * :func:`read_document_term_matrix()` reads a document-term matrix from a CSV file.
    * :func:`read_files()` reads one or multiple files based on a pathlist.
    * :func:`remove_features()` removes features from a ``document_term_matrix``.
    * :func:`segment()` is a wrapper for :func:`segment_fuzzy()` and segments a \
    ``tokenized_document`` into segments of a certain number of tokens, respecting existing chunks.
    * :func:`segment_fuzzy()` segments a ``tokenized_document``, tolerating existing \
    chunks (like paragraphs).
    * :func:`split_paragraphs()` splits a ``document`` or ``dkpro_document`` by paragraphs.
    * :func:`tokenize()` tokenizes a ``document`` based on a Unicode regular expression.
"""


from collections import Counter, defaultdict
import csv
from itertools import chain
import os
from lxml import etree
import numpy as np
import pandas as pd
import pickle
import regex
import logging


log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
logging.basicConfig(level=logging.WARNING,
                    format='%(levelname)s %(name)s: %(message)s')


def _create_bag_of_words(document_labels, tokenized_corpus):
    """Creates a bag-of-words model.

    This private function is wrapped in :func:`_create_large_corpus_model()`. The \
    first level consists of the document label as key, and the dictionary \
    of counts as value. The second level consists of type ID as key, and the \
    count of types in document pairs as value.

    Args:
        document_labels (list): Iterable of document labels.
        tokenized_corpus (list): Tokenized corpus as an iterable
            containing one or more iterables containing tokens.

    Returns:
        A bag-of-words model as dictionary of dictionaries, document IDs and type IDs.

    Example:
        >>> document_labels = ['exampletext']
        >>> tokenized_corpus = [['this', 'is', 'an', 'example', 'text']]
        >>> bag_of_words, document_ids, type_ids = _create_bag_of_words(document_labels, tokenized_corpus)
        >>> isinstance(bag_of_words, dict) and isinstance(document_ids, dict) and isinstance(type_ids, dict)
        True
    """
    document_ids = _token2id(document_labels)
    type_ids = _token2id(tokenized_corpus)
    bag_of_words = defaultdict(dict)
    for document_label, tokenized_document in zip(document_labels, tokenized_corpus):
        bag_of_words[document_label] = Counter([type_ids[token] for token in tokenized_document])
    return {document_ids[id_]: doc for id_, doc in bag_of_words.items()}, document_ids, type_ids


def _create_large_corpus_model(tokenized_corpus, document_labels):
    """Creates a document-term matrix for large corpora.

    This private function is wrapped in :func:`create_document_term_matrix()` and \
    creates a pandas DataFrame containing document and type IDs as MultiIndex \
    and type frequencies as values representing the counts of tokens for each \
    token in each document.

    Args:
        tokenized_corpus (list): Tokenized corpus as an iterable
            containing one or more iterables containing tokens.
        document_labels (list): Iterable of document labels.

    Returns:
        A document-term matrix as pandas DataFrame, ``document_ids`` and ``type_ids``.

    Todo:
        * Make the whole function faster.

    Example:
        >>> tokenized_corpus = [['this', 'is', 'document', 'one'], ['this', 'is', 'document', 'two']]
        >>> document_labels = ['document_one', 'document_two']
        >>> document_term_matrix, document_ids, type_ids = _create_large_corpus_model(tokenized_corpus, document_labels)
        >>> isinstance(document_term_matrix, pd.DataFrame) and isinstance(document_ids, dict) and isinstance(type_ids, dict)
        True
    """
    bag_of_words, document_ids, type_ids = _create_bag_of_words(document_labels, tokenized_corpus)
    multi_index = _create_multi_index(bag_of_words)
    document_term_matrix = pd.DataFrame(np.zeros((len(multi_index), 1), dtype=int), index=multi_index)
    index_iterator = multi_index.groupby(multi_index.get_level_values('document_id'))

    for document_id in range(1, len(multi_index.levels[0]) + 1):
        for type_id in [val[1] for val in index_iterator[document_id]]:
            document_term_matrix.at[(document_id, type_id), 0] = int(bag_of_words[document_id][type_id])
    return document_term_matrix, document_ids, type_ids


def _create_multi_index(bag_of_words):
    """Creates a MultiIndex for a pandas DataFrame.

    This private function is wrapped in :func:`_create_large_corpus_model()`.

    Args:
        bag_of_words (dict): A bag-of-words model of ``{document_id: {type_id: frequency}}``.

    Returns:
        Pandas MultiIndex.

    Example:
        >>> bag_of_words = {1: {1: 2, 2: 3, 3: 4}}
        >>> _create_multi_index(bag_of_words)
        MultiIndex(levels=[[1], [1, 2, 3]],
                   labels=[[0, 0, 0], [0, 1, 2]],
                   names=['document_id', 'type_id'])
    """
    tuples = []
    for document_id in range(1, len(bag_of_words) + 1):
        if len(bag_of_words[document_id]) == 0:
            tuples.append((document_id, 0))
        for type_id in bag_of_words[document_id]:
            tuples.append((document_id, type_id))
    return pd.MultiIndex.from_tuples(tuples, names=['document_id', 'type_id'])



def _hapax_legomena_large_corpus_model(document_term_matrix, type_ids):
    """Determines hapax legomena in large corpus model.

    This private function is wrapped in :func:`find_hapax_legomena()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.

    Returns:
        Hapax legomena in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> find_hapax_legomena(document_term_matrix, type_ids)
        ['hapax']
    """
    id2type = {id_: type_ for type_, id_ in type_ids.items()}
    document_term_matrix_collapsed = document_term_matrix.groupby(document_term_matrix.index.get_level_values('type_id')).sum()
    hapax_legomena = document_term_matrix_collapsed.loc[document_term_matrix_collapsed[0] == 1]
    return [id2type[token] for token in hapax_legomena.index.get_level_values('type_id')]


def _remove_features_from_large_corpus_model(document_term_matrix, type_ids, features):
    """Removes features from large corpus model.

    This private function is wrapped in :func:`remove_features()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.
        features (list): A list of tokens.

    Returns:
        A clean document-term matrix as pandas DataFrame.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['token', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> len(_remove_features_from_large_corpus_model(document_term_matrix, type_ids, ['token']))
        1
    """
    features = [token for token in set(type_ids.keys()) if token in features]
    return document_term_matrix.drop([type_ids[token] for token in features], level='type_id')





def _list_mfw_large_corpus_model(document_term_matrix, type_ids, most_frequent_tokens):
    """Determines stopwords in large corpus model.

    This private function is wrapped in :func:`list_mfw()`.

    Args:
        document_term_matrix (pandas.DataFrame): A document-term matrix.
        type_ids (dict): A dictionary with types as key and identifiers as values.
        most_frequent_tokens (int, optional): Treshold for most frequent tokens.

    Returns:
        Most frequent tokens in a list.

    Example:
        >>> document_labels = ['document']
        >>> tokenized_corpus = [['hapax', 'stopword', 'stopword']]
        >>> document_term_matrix, _, type_ids = create_document_term_matrix(tokenized_corpus, document_labels, large_corpus=True)
        >>> list_mfw(document_term_matrix, 1, type_ids)
        ['stopword']
    """
    id2type = {id_: type_ for type_, id_ in type_ids.items()}
    document_term_matrix_collapsed = document_term_matrix.groupby(document_term_matrix.index.get_level_values('type_id')).sum()
    stopwords = document_term_matrix_collapsed[0].nlargest(most_frequent_tokens)
    return [id2type[token] for token in stopwords.index.get_level_values('type_id')]
