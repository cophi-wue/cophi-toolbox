import pytest
import pathlib
from cophi.text import model, utils


DOCUMENT = "A B C D E F"

def make_file(tmpdir, fname, content):
    p = tmpdir.mkdir("sub").join(fname)
    p.write(content)
    return p

@pytest.fixture
def document():
    return model.Document(DOCUMENT, "document", r"\w")

@pytest.fixture
def corpus(document):
    return model.Corpus([document])

def test_document(tmpdir):
    filepath = make_file(tmpdir, "document.txt", DOCUMENT)
    document = cophi.document(str(filepath), token_pattern=r"\w")
    assert document.text == DOCUMENT

def test_corpus(tmpdir):
    p = make_file(tmpdir, "document.txt", DOCUMENT)
    directory = pathlib.Path(str(p)).parent
    corpus, metadata = cophi.corpus(directory, metadata=True)
    assert metadata["parent"].iloc[0] == str(directory)
    assert corpus.documents[0].text == DOCUMENT

def test_export(corpus):
    output = pathlib.Path("corpus.svmlight")
    utils.export(corpus.dtm, output, "svmlight")
    assert output.exists()
    with output.open("r", encoding="utf-8") as file:
        assert file.read() == "document document a:1 b:1 c:1 d:1 e:1 f:1\n"

    output = pathlib.Path("corpus.txt")
    utils.export(corpus.dtm, output, "text")
    assert output.exists()
    with output.open("r", encoding="utf-8") as file:
        assert file.read() == "document document a b c d e f\n"

    with pytest.raises(ValueError):
        utils.export(corpus.dtm, output, "unknown")
