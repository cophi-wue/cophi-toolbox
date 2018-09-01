import pytest
import pathlib
import cophi


DOCUMENT = "A B C D E F"

def make_file(tmpdir, fname, content):
    p = tmpdir.mkdir("sub").join(fname)
    p.write(content)
    return p

def test_document(tmpdir):
    filepath = make_file(tmpdir, "document.txt", DOCUMENT)
    document = cophi.document(str(filepath), token_pattern=r"\w")
    assert document.text == DOCUMENT

def test_corpus(tmpdir):
    p = make_file(tmpdir, "document.txt", DOCUMENT)
    directory = pathlib.Path(str(p)).parent
    corpus, metadata = cophi.corpus(directory)
    assert metadata["parent"].iloc[0] == str(directory)
    assert corpus.documents[0].text == DOCUMENT
