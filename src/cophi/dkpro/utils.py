"""
cophi.dkpro.utils
~~~~~~~~~~~~~~~~~

This module implements general helper functions.
"""

import logging
import subprocess

from cophi import dkpro, text


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def call(args: list) -> None:
    """Call a subprocess.

    Parameter:
        args (list): The subprocess’ arguments.
    """
    for message in _process(args):
        logger.info(message)


def _process(args: list) -> None:
    """Construct a process.

    Parameter:
        args (list): The subprocess’ arguments.
    """
    popen = subprocess.Popen(args,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
    # Yield every line of stdout:
    for line in iter(popen.stdout.readline, ""):
        yield line.strip()
    popen.stdout.close()
    code = popen.wait()
    if code:
        raise subprocess.CalledProcessError(code, args)
