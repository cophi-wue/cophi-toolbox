"""
cophi.dkpro.api
~~~~~~~~~~~~~~~

This module implements the high-level API for the DARIAH-DKPro-Wrapper.
"""

from pathlib import Path
import tempfile

from cophi import dkpro, text


def process(path, jar, language, reader, xms="4g"):
    """Process a textfile with the DARIAH-DKPro-Wrapper.

    Parameters:
        path (str): Path to text file.
        jar (str): Path to JAR file.
        language (str): Language of the text.
        reader (str): File reader, either `text` or `xml`.
        xms (str): Size to allocate by JVM.
    """
    output = Path(tempfile.gettempdir(), "dariah-dkpro-output")
    if not output.exists():
        output.mkdir()

    d = dkpro.model.DKPro(jar=jar,
                          xms=xms)

    d.process(input=path,
              output=output,
              language=language,
              reader=reader)

    for file in output.glob("*.csv"):
        yield dkpro.model.Document(file)


def pipe(filepath, jar, language, lemma, pos, **kwargs):
    """Pipe a file through DARIAH-DKPro-Wrapper.
    """
    for doc in process(filepath, jar, language, "text"):
        if pos:
            doc = doc.filter(pos)
            title = doc.name
        else:
            doc = doc.raw
            title = doc.name
        content = " ".join(doc["Lemma" if lemma else "Token"])
        return text.model.Document(content, title=title, **kwargs)
