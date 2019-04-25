"""
text.model
~~~~~~~~~~~~~~~~

This module implements low-level model classes.
"""

import collections
import itertools
import logging
import math
import pathlib

import lxml.etree
import numpy as np
import pandas as pd
import regex as re

from cophi.text import utils, complexity


logger = logging.getLogger(__name__)


class Textfile:
    """Model class for a Textfile.

    Parameters:
        filepath (str): Path to a text file.
        treat_as (str): Treat text file like .txt or .xml (optional).
        encoding (str): Encoding to use for UTF when reading (optional).

    Attributes:
        filepath (pathlib.Path): Text file’s Path object.
        title (str): Filename without parent or suffix.
        suffix (str): Text file’s extension.
        treat_as (str) Treated text file like this suffix.
        parent (str): Parent path of text file.
        encoding (str): Encoding used for UTF when reading.
    """

    def __init__(self, filepath, treat_as=None, encoding="utf-8"):
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        self.filepath = filepath
        self.title = self.filepath.stem
        self.suffix = self.filepath.suffix
        self.parent = str(self.filepath.parent)
        self.encoding = encoding
        if treat_as is not None and treat_as not in {".txt", ".xml"}:
            raise ValueError("The file format '{}' is not supported. "
                             "Try '.txt', or '.xml'.".format(treat_as))
        else:
            self.treat_as = treat_as

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        pass

    def parse_xml(self, parser=lxml.etree.XMLParser()):
        """Parse an XML file.

        Parameters:
            parser: XML parser object.

        Returns:
            An :class:`etree._ElemenTree` object.
        """
        return lxml.etree.parse(str(self.filepath), parser=parser)

    @staticmethod
    def stringify(tree):
        """Serialize to an encoded string representation of its XML tree.

        Parameters:
            tree: An :class:`etree._ElemenTree`.
            encoding: Encoding to use when serializing.
        """
        return lxml.etree.tostring(tree, method="text", encoding=str)

    @property
    def content(self):
        """Content of text file.
        """
        if (self.treat_as is None and self.suffix == ".txt") or (self.treat_as == ".txt"):
            return self.filepath.read_text(encoding=self.encoding)
        elif (self.treat_as is None and self.suffix == ".xml") or (self.treat_as == ".xml"):
            tree = self.parse_xml()
            return self.stringify(tree)

    @property
    def size(self):
        """Size of text file content in characters.
        """
        return len(self.content)


class Document:
    """Model class for a Document.

    Parameters:
        text (str): Content of a text file.
        title (str): Describing title for the document (optional).
        lowercase (bool): If True, writes all letters in lowercase (optional).
        n (int): Number of tokens per ngram (optional).
        token_pattern (str): Regex pattern for one token (optional).
        maximum (int): Stop tokenizing after that much tokens (optional).

    Attributes:
        text (str): Content of the text file.
        title (str): Describing title for the document.
        lowercase (bool): If True, all letters are lowercase.
        n (int): Number of words per ngram.
        token_pattern (str): Regex pattern for one token.
        maximum (int): Stopped tokenizing after that much tokens.
        tokens (list): Tokenized content of the document.
    """

    def __init__(self, text, title=None,
                 token_pattern=r"\p{L}+\p{Connector_Punctuation}?\p{L}+",
                 lowercase=True, n=None, maximum=None):
        self.text = text
        self.title = title
        self.lowercase = lowercase
        if n is not None and n < 1:
            raise ValueError("Arg 'n' must be greater than {}.".format(n))
        self.n = n
        self.token_pattern = token_pattern
        self.maximum = maximum
        self.tokens = list(utils.find_tokens(self.text,
                                             self.token_pattern,
                                             self.maximum))
        if self.lowercase:
            self.tokens = utils.lowercase_tokens(self.tokens)

    @property
    def ngrams(self):
        """Constructed ngrams.
        """
        try:
            if self.n > 1:
                return utils.construct_ngrams(self.tokens, self.n)
            else:
                return self.tokens
        except TypeError:
            raise ValueError("You did not set a value for ngrams.")

    @property
    def types(self):
        """Document vocabulary.
        """
        return set(self.tokens)

    @property
    def lengths(self):
        """Token lengths.
        """
        return np.array([len(token) for token in self.tokens])

    @property
    def mean_length(self):
        """Arithmetic mean of token lengths.
        """
        return self.lengths.mean()

    @property
    def num_tokens(self):
        """Number of tokens.
        """
        return len(self.tokens)

    @property
    def num_types(self):
        """Number of types.
        """
        return len(self.types)

    @property
    def bow(self):
        """Bag-of-words representation.
        """
        return pd.Series(collections.Counter(self.tokens))

    @property
    def rel(self):
        """Bag-of-words representation with relative frequencies.
        """
        return self.bow / self.num_tokens

    def mfw(self, n=10, rel=False, as_list=True):
        """Most frequent words.

        Parameters:
            n (int): Number of most frequent words (optional).
            rel (bool): If True, use relative frequencies for
                sorting (optional).
            as_list (bool): If True, return just tokens in a
                list (optional).
        """
        if rel:
            freqs = self.rel.sort_values(ascending=False).iloc[:n]
        else:
            freqs = self.bow.sort_values(ascending=False).iloc[:n]
        if as_list:
            return list(freqs.index)
        else:
            return freqs

    @property
    def hapax(self):
        """Hapax legomena.
        """
        freqs = self.bow
        return list(freqs[freqs == 1].index)

    def window(self, size=1000):
        """Iterate with a sliding window over tokens.

        Parameters:
            size (int): Window size in tokens (optional).
        """
        for i in range(int(self.num_tokens / size)):
            yield self.tokens[i * size:(i * size) + size]

    @property
    def freq_spectrum(self):
        """Frequency spectrum.
        """
        bow = collections.Counter(self.tokens)  # no pandas needed here
        return pd.Series(collections.Counter(bow.values()))

    @staticmethod
    def drop(tokens, features):
        """Drop features.

        Parameters:
            tokens (list): Tokenized document.
            features (list): Features to drop drom tokenized document.
        """
        return (token for token in tokens if token not in features)

    def paragraphs(self, sep=r"\n"):
        """Paragraphs as separate entities.

        Parameters:
            sep (str): Pattern which indicates a paragraph.
        """
        if not hasattr(sep, "match"):
            sep = re.compile(sep)
        splitted = sep.split(self.text)
        return filter(None, splitted)

    def segments(self, size=1000, tolerance=0.05, flatten=True):
        """Segments as separate entities.

        Parameters:
            size (int): Size of one segment (optional).
            tolerance (float): Threshold value for respecting
                paragraph borders (optional).
            flatten (bool): If True, flatten the segments list (optional).
        """
        segments = utils.segment_fuzzy([self.tokens],
                                             size,
                                             tolerance)
        if flatten:
            if not callable(flatten):
                def flatten_chunks(segment):
                    return list(itertools.chain.from_iterable(segment))
            segments = map(flatten_chunks, segments)
        return segments

    def bootstrap(self, measure="ttr", window=1000, **kwargs):
        """Calculate complexity with a sliding window.

        Parameters:
            measure (str): Measure to use, possible values are
                'ttr', 'guiraud_r', 'herdan_c', 'dugast_k',
                'maas_a2', 'dugast_u', 'tuldava_ln', 'brunet_w',
                'cttr', 'summer_s', 'honore_h', 'sichel_s',
                'michea_m', 'entropy', 'yule_k', 'simpsons_d',
                'herdan_vm', or 'orlov_z'.
            window (int): Size of sliding window (optional).
            **kwargs: Additional parameters for
                :func:`complexity.orlov_z` (optional).
        """
        for chunk in self.window(window):
            parameter = utils._parameter(chunk, measure)
            calculate_complexity = complexity.wrapper(measure)
            yield calculate_complexity(**parameter, **kwargs)

    def complexity(self, measure="ttr", window=None, **kwargs):
        """Calculate complexity, optionally with a sliding window.

        Parameters:
            measure (str): Measure to use, possible values are
                'ttr', 'guiraud_r', 'herdan_c', 'dugast_k',
                'maas_a2', 'dugast_u', 'tuldava_ln', 'brunet_w',
                'cttr', 'summer_s', 'honore_h', 'sichel_s',
                'michea_m', 'entropy', 'yule_k', 'simpsons_d',
                'herdan_vm', or 'orlov_z'.
            window (int): Size of sliding window (optional).
            **kwargs: Additional parameters for
                :func:`complexity.orlov_z` (optional).
        """
        if measure == "ttr":
            if window:
                sttr = list(self.bootstrap(measure, window))
                return np.array(sttr).mean(), complexity.ci(sttr)
            else:
                calculate_complexity = complexity.wrapper(measure)
                parameter = utils._parameter(self.tokens, measure)
                return calculate_complexity(**parameter)
        elif measure == "orlov_z":
            if window:
                orlov = list(self.bootstrap(measure, window, **kwargs))
                return np.array(orlov).mean()
            else:
                calculate_complexity = complexity.wrapper(measure)
                parameter = utils._parameter(self.tokens, measure)
                return calculate_complexity(**parameter, **kwargs)
        else:
            if window:
                results = list(self.bootstrap(measure, window))
                return np.array(results).mean()
            else:
                calculate_complexity = complexity.wrapper(measure)
                parameter = utils._parameter(self.tokens, measure)
                return calculate_complexity(**parameter)


class Corpus:
    """Model class for a Corpus.

    Parameters:
        documents (iterable): One or more Document objects.
        sparse (str): If True, use the sparse DataFrame. NOT IMPLEMENTED.

    Attributes:
        dtm (pd.DataFrame): Document-term matrix with absolute
            word frequencies.
    """

    def __init__(self, documents, sparse=False):
        if sparse:
            raise NotImplementedError("This feature is not yet "
                                      "implemented. If you wish "
                                      "to use sparse matrices "
                                      "(because you have a very "
                                      "large corpus), feel free "
                                      "to create a new issue on "
                                      "GitHub.")
        else:
            matrix = pd.DataFrame
        self.documents = documents

        def count_corpus(documents):
            corpus = dict()
            for document in documents:
                logger.info("Processing '{}'...".format(document.title))
                corpus[document.title] = document.bow
            return corpus
        counts = count_corpus(self.documents)
        logger.info("Constructing document-term matrix...")
        self.dtm = matrix(counts)
        self.dtm = self.dtm.T

    @staticmethod
    def map_metadata(data, metadata, uuid="uuid", fields=["title"], sep="_"):
        """Map metadata with a UUID.

        Parameters:
            data: Data (e.g. a pandas DataFrame) to map with.
            metadata: Matrix with metadata, one row corresponds
                to one document.
            uuid (str): The connecting UUID between `data`
                and `metadata` (optional).
            fields (list): One or more columns of `metadata` (optional).
            sep (str): Glue multiple `fields` with this
                separator together (optional).
        """
        data = data.copy()  # do not work on original object itself
        document_uuid = metadata[uuid]
        index = metadata[fields[0]].astype(str)
        if len(fields) > 1:
            for field in fields[1:]:
                index = index + sep + metadata[field].astype(str)
        document_uuid.index = index
        data.index = document_uuid.to_dict()
        return data

    @property
    def stats(self):
        """Corpus statistics, e.g. number of documents.
        """
        s = pd.Series(self.dtm.shape, index=["documents", "types"])
        s["tokens"] = self.num_tokens.sum()
        s["hapax"] = len(self.hapax)
        return s

    @property
    def freq_spectrum(self):
        """Frequency spectrum of types.
        """
        return self.dtm.sum(axis=0).value_counts()

    @property
    def types(self):
        """Corpus vocabulary.
        """
        return list(self.dtm.columns)

    @staticmethod
    def sort(dtm):
        """Descending sorted document-term matrix.

        Parameters:
            dtm (pd.DataFrame): Document-term matrix.
        """
        return dtm.iloc[:, (-dtm.sum()).argsort()]

    def mfw(self, n=100, rel=False, as_list=True):
        """Most frequent words.

        Parameters:
            n (int): Number of most frequent words (optional).
            rel (int): If True, use relative frequencies for
                sorting (optional).
            as_list: If True, return just tokens in a list (optional).
        """
        dtm = self.sort(self.dtm)
        if rel:
            mfw = dtm.iloc[:, :n].div(self.dtm.sum(axis=1), axis=0)
        else:
            mfw = dtm.iloc[:, :n]
        if as_list:
            return list(mfw.columns)
        else:
            return mfw.sum()

    @property
    def hapax(self):
        """Hapax legomena.
        """
        return list(self.dtm.loc[:, self.dtm.max() == 1].columns)

    @staticmethod
    def drop(dtm, features):
        """Drop features from document-term matrix.

        Parameters:
            dtm (pd.DataFrame): Document-term matrix.
            features (iterable): Types to drop from document-term matrix.
        """
        features = [token for token in features if token in dtm.columns]
        return dtm.drop(features, axis=1)

    @staticmethod
    def cull(dtm, ratio=None, threshold=None, keepna=False):
        """Remove features that do not appear in a minimum of documents.

        Parameters:
            dtm (pd.DataFrame): Document-term matrix.
            ratio (float): Minimum ratio of documents a word must occur in.
            threshold (int): Minimum number of documents a word must occur in.
            keepna (bool): If True, keep missing words as NaN instead of 0.
        """
        if ratio is not None:
            if ratio > 1:
                threshold = ratio
            else:
                threshold = math.ceil(ratio * dtm.index.size)
        elif threshold is None:
            return dtm

        culled = dtm.replace(0, np.nan).dropna(thresh=threshold, axis=1)
        if not keepna:
            culled = culled.fillna(0)
        return culled

    @property
    def zscores(self):
        """Standardized document-term matrix.

        Used formula:
        .. math::
            z_x = \frac{x - \mu}{\sigma}
        """
        return (self.dtm - self.dtm.mean()) / self.dtm.std()

    @property
    def rel(self):
        """Document-term matrix with relative word frequencies.
        """
        return self.dtm.div(self.dtm.sum(axis=1), axis=0)

    @property
    def tfidf(self):
        """TF-IDF normalized document-term matrix.

        Used formula:
        .. math::
            tf-idf_{t,d} = tf_{t,d} \times idf_t = \
            tf_{t,d} \times log(\frac{N}{df_t})
        """
        tf = self.rel
        idf = self.stats["documents"] / self.dtm.fillna(0).astype(bool).sum(axis=0)
        return tf * np.log(idf)

    @property
    def num_types(self):
        """Number of types.
        """
        return self.dtm.replace(0, np.nan).count(axis=1)

    @property
    def num_tokens(self):
        """Number of tokens.
        """
        return self.dtm.sum(axis=1)

    def complexity(self, window=1000, measure="ttr"):
        """Calculate complexity for each document with a sliding window.

        Parameters:
            measure (str): Measure to use, possible values are
                'ttr', 'guiraud_r', 'herdan_c', 'dugast_k',
                'maas_a2', 'dugast_u', 'tuldava_ln', 'brunet_w',
                'cttr', 'summer_s', 'honore_h', 'sichel_s',
                'michea_m', 'entropy', 'yule_k', 'simpsons_d',
                'herdan_vm', or 'orlov_z'.
            window (int): Size of sliding window (optional).
            **kwargs: Additional parameters for
                :func:`complexity.orlov_z` (optional).
        """
        if measure == "ttr":
            results = pd.DataFrame()
        else:
            results = pd.Series()
        for document in self.documents:
            if measure == "ttr":
                sttr, ci = document.complexity(measure, window)
                results = results.append(pd.DataFrame({"sttr": sttr,
                                                       "ci": ci},
                                                      index=[document.title]))
            else:
                results[document.title] = document.complexity(measure, window)
        return results

    @property
    def ttr(self):
        """Type-Token Ratio (TTR).
        """
        return complexity.ttr(self.num_types.sum(),
                                    self.num_tokens.sum())

    @property
    def guiraud_r(self):
        """Guiraud’s R (1954).
        """
        return complexity.guiraud_r(self.num_types.sum(),
                                          self.num_tokens.sum())

    @property
    def herdan_c(self):
        """Herdan’s C (1960, 1964).
        """
        return complexity.herdan_c(self.num_types.sum(),
                                         self.num_tokens.sum())

    @property
    def dugast_k(self):
        """Dugast’s k (1979).
        """
        return complexity.dugast_k(self.num_types.sum(),
                                         self.num_tokens.sum())

    @property
    def dugast_u(self):
        """Dugast’s U (1978, 1979).
        """
        return complexity.dugast_k(self.num_types.sum(),
                                         self.num_tokens.sum())

    @property
    def maas_a2(self):
        """Maas’ a^2 (1972).
        """
        return complexity.maas_a2(self.num_types.sum(),
                                        self.num_tokens.sum())

    @property
    def tuldava_ln(self):
        """Tuldava’s LN (1977).
        """
        return complexity.tuldava_ln(self.num_types.sum(),
                                           self.num_tokens.sum())

    @property
    def brunet_w(self):
        """Brunet’s W (1978).
        """
        return complexity.brunet_w(self.num_types.sum(),
                                         self.num_tokens.sum())

    @property
    def cttr(self):
        """Carroll’s Corrected Type-Token Ratio (CTTR).
        """
        return complexity.cttr(self.num_types.sum(),
                                     self.num_tokens.sum())

    @property
    def summer_s(self):
        """Summer’s S.
        """
        return complexity.summer_s(self.num_types.sum(),
                                         self.num_tokens.sum())

    @property
    def sichel_s(self):
        """Sichel’s S (1975).
        """
        return complexity.sichel_s(self.num_types.sum(),
                                         self.freq_spectrum)

    @property
    def michea_m(self):
        """Michéa’s M (1969, 1971).
        """
        return complexity.michea_m(self.num_types.sum(),
                                         self.freq_spectrum)

    @property
    def honore_h(self):
        """Honoré's H (1979).
        """
        return complexity.honore_h(self.num_types.sum(),
                                         self.num_tokens.sum(),
                                         self.freq_spectrum)

    @property
    def entropy(self):
        """Entropy S.
        """
        return complexity.entropy(self.num_tokens.sum(),
                                        self.freq_spectrum)

    @property
    def yule_k(self):
        """Yule’s K (1944).
        """
        return complexity.yule_k(self.num_tokens.sum(),
                                       self.freq_spectrum)

    @property
    def simpson_d(self):
        """Simpson’s D (1949).
        """
        return complexity.simpson_d(self.num_tokens.sum(),
                                          self.freq_spectrum)

    @property
    def herdan_vm(self):
        """Herdan’s VM (1955).
        """
        return complexity.herdan_vm(self.num_types.sum(),
                                          self.num_tokens.sum(),
                                          self.freq_spectrum)

    def orlov_z(self, max_iterations=100, min_tolerance=1):
        """Orlov’s Z (1983).
        """
        return complexity.orlov_z(self.num_tokens.sum(),
                                        self.num_types.sum(),
                                        self.freq_spectrum,
                                        max_iterations,
                                        min_tolerance)

    @classmethod
    def svmlight(cls, dtm, filepath):
        """Export corpus to SVMLight format.

        Parameters:
            dtm: Document-term matrix.
            filepath: Path to output file.
        """
        with pathlib.Path(filepath).open("w", encoding="utf-8") as file:
            for title, document in dtm.iterrows():
                # Drop types with zero frequencies:
                document = document.dropna()
                features = ["{word}:{freq}".format(word=word, freq=int(
                    freq)) for word, freq in document.iteritems()]
                export = "{title} {features}\n".format(
                    title=title, features=" ".join(features))
                file.write(export)

    @classmethod
    def plaintext(cls, dtm, filepath):
        """Export corpus to plain text format.

        Parameters:
            dtm: Document-term matrix.
            filepath: Path to output file.
        """
        with pathlib.Path(filepath).open("w", encoding="utf-8") as file:
            for title, document in dtm.iterrows():
                # Drop types with zero frequencies:
                document = document.dropna()
                features = [" ".join([word] * int(freq))
                            for word, freq in document.iteritems()]
                export = "{title} {features}\n".format(
                    title=title, features=" ".join(features))
                file.write(export)


class Metadata(pd.DataFrame):
    """Handle corpus metadata.

    Feel free to implement some fancy stuff here.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
