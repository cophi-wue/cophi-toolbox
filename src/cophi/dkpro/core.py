"""
cophi.dkpro.core
~~~~~~~~~~~~~~~~

This module implements the core functions of the DKPro module.
"""

import csv
from pathlib import Path

import pandas as pd

from cophi import dkpro


def call(jar, xms="4g", **parameters):
    """Call DARIAH DKPro-Wrapper.

    Parameter:
        xms (str): Initial memory allocation pool for Java Virtual Machine.
        jar (str): Path to jarfile.
        **parameter: Additional parameters for DARIAH DKPro-Wrapper.
    """
    # Basic subprocess command:
    args = ["java", "-Xms{}".format(xms), "-jar", jar]

    # Append additional parameters:
    for parameter, value in parameters.items():
        # Support synonyms for `-input` parameter:
        if parameter in {"filepath", "directory", "path", "corpus"}:
            args.append("-input")
        else:
            args.append("-{}".format(parameter))
        if value:
            args.append(str(value))
    return dkpro.utils.call(args)
