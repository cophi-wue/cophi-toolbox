"""
cophi.dkpro.model
~~~~~~~~~~~~~~~~~

This module implements model classes for the DARIAH-DKPro Wrapper.
"""

import csv
from pathlib import Path

import pandas as pd

from cophi import dkpro, text


class DKPro:
    """DARIAH DKPro-Wrapper.
    """
    def __init__(self, jar, xms="4g"):
        self.jar = jar
        self.xms = xms

    def process(self, **parameters):
        """Process a single text file or a whole directory.

        Parameters:
            path (str): Path to text file or directory.
            config (str): Config file.
            language (str): Corpus language code. Defaults to `en`.
            output (str): Path to output directory.
            reader (str): Either `text` (default) or `xml`.

        Returns:
            True, if call was successful.
        """
        return dkpro.core.call(self.jar, self.xms, **parameters)


class Document:
    def __init__(self, filepath):
        self.filepath = Path(filepath)

    @property
    def raw(self):
        document = pd.read_csv(self.filepath,
                               sep="\t",
                               quoting=csv.QUOTE_NONE)
        filename = Path(self.filepath.stem)
        document.name = filename.stem
        return document

    def filter(self, pos):
        document = self.raw[self.raw["CPOS"].isin(pos)]
        document.name = self.raw.name
        return document
