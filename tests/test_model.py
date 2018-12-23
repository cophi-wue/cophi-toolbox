import pathlib

import pytest
import lxml
import numpy as np
import pandas as pd
import cophi


DOCUMENT = "AAABBCCCDEF"
TOKENS = list(DOCUMENT)
LOWERCASE_TOKENS =  [token.lower() for token in TOKENS]

def make_file(tmpdir, fname, content):
    p = tmpdir.mkdir("sub").join(fname)
    p.write(content)
    return p

@pytest.fixture
def textfile_suffix(tmpdir):
    p = make_file(tmpdir, "document.txt", DOCUMENT)
    return cophi.model.Textfile(str(p), treat_as=None)

@pytest.fixture
def textfile_txt(tmpdir):
    p = make_file(tmpdir, "document.txt", DOCUMENT)
    return cophi.model.Textfile(str(p), treat_as=".txt")

@pytest.fixture
def textfile_xml(tmpdir):
    p = make_file(tmpdir, "document.xml", "<xml>{}</xml>".format(DOCUMENT))
    return cophi.model.Textfile(str(p), treat_as=".xml")

@pytest.fixture
def document():
    return cophi.model.Document(DOCUMENT, "document", r"\w")

@pytest.fixture
def corpus(document):
    return cophi.model.Corpus([document])


class TestTextfile:
    def test_suffix(self, textfile_suffix, tmpdir):
        assert str(textfile_suffix.filepath) == str(tmpdir.join("sub", "document.txt"))
        assert textfile_suffix.title == "document"
        assert textfile_suffix.suffix == ".txt"
        assert textfile_suffix.parent == str(tmpdir.join("sub"))
        assert textfile_suffix.encoding == "utf-8"
        assert textfile_suffix.treat_as == None
        assert textfile_suffix.content == DOCUMENT
        assert textfile_suffix.size == len(DOCUMENT)

    def test_txt(self, textfile_txt, tmpdir):
        assert str(textfile_txt.filepath) == str(tmpdir.join("sub", "document.txt"))
        assert textfile_txt.title == "document"
        assert textfile_txt.suffix == ".txt"
        assert textfile_txt.parent == str(tmpdir.join("sub"))
        assert textfile_txt.encoding == "utf-8"
        assert textfile_txt.treat_as == ".txt"
        assert textfile_txt.content == DOCUMENT
        assert textfile_txt.size == len(DOCUMENT)

    def test_xml(self, textfile_xml, tmpdir):
        assert str(textfile_xml.filepath) == str(tmpdir.join("sub", "document.xml"))
        assert textfile_xml.title == "document"
        assert textfile_xml.suffix == ".xml"
        assert textfile_xml.parent == str(tmpdir.join("sub"))
        assert textfile_xml.encoding == "utf-8"
        assert textfile_xml.treat_as == ".xml"
        assert textfile_xml.content == DOCUMENT
        assert textfile_xml.size == len(DOCUMENT)

    def test_parse_xml(self, textfile_xml):
        assert isinstance(textfile_xml.parse_xml(), lxml.etree._ElementTree)

    def test_stringify(self, textfile_xml):
        tree = textfile_xml.parse_xml()
        assert textfile_xml.stringify(tree) == "AAABBCCCDEF"

    def test_value_error(self, tmpdir):
        with pytest.raises(ValueError):
            cophi.model.Textfile("raises", treat_as="error")


class TestDocument:
    def test_attributes(self, document):
        assert document.text == DOCUMENT
        assert document.title == "document"
        assert document.lowercase == True
        assert document.n == None
        assert document.token_pattern == r"\w"
        assert document.maximum == None
        assert document.tokens == LOWERCASE_TOKENS

    def test_ngram_value_error(self):
        with pytest.raises(ValueError):
            cophi.model.Document(DOCUMENT, n=0)

    def test_ngrams(self):
        document = cophi.model.Document(DOCUMENT, token_pattern=r"\w", n=2)
        assert list(document.ngrams)[0] == "a a"
        document = cophi.model.Document(DOCUMENT, token_pattern=r"\w", n=1)
        assert document.ngrams == LOWERCASE_TOKENS
        with pytest.raises(ValueError):
            document = cophi.model.Document(DOCUMENT, token_pattern=r"\w", n=None)
            document.ngrams == LOWERCASE_TOKENS

    def test_types(self, document):
        assert len(document.types) == len(set(TOKENS))

    def test_lengths(self, document):
        assert len(document.lengths) == len(TOKENS)

    def test_mean_length(self, document):
        assert document.mean_length == 1

    def test_num_tokens(self, document):
        assert document.num_tokens == len(TOKENS)

    def test_num_types(self, document):
        assert document.num_types == len(set(TOKENS))

    def test_bow(self, document):
        assert document.bow.sum() == 11

    def test_rel(self, document):
        assert round(document.rel.sum()) == 1

    def test_mfw(self, document):
        assert document.mfw(1, as_list=False).sum() == 3
        assert round(document.mfw(1, rel=True, as_list=False).sum()) == 0
        assert len(document.mfw(1, as_list=True)) == 1

    def test_hapax(self, document):
        assert document.hapax == ["d", "e", "f"]

    def test_window(self, document):
        for expected, chunk in zip(LOWERCASE_TOKENS, document.window(1)):
            assert expected == chunk[0]

    def test_freq_spectrum(self, document):
        assert document.freq_spectrum.sum() == 6

    def test_drop(self, document):
        features = ["a", "b", "c"]
        tokens = document.drop(LOWERCASE_TOKENS, features)
        assert list(tokens) == ["d", "e", "f"]

    def test_paragraphs(self, document):
        assert list(document.paragraphs()) == [DOCUMENT]

    def test_segments(self, document):
        assert list(document.segments(1))[0] == ["a"]
        assert list(document.segments(1, flatten=False))[0] == [["a"]]

    def test_bootstrap(self, document):
        assert list(document.bootstrap(window=5)) == [0.4, 0.6]

    def test_complexity(self, document):
        ttr = document.complexity("ttr")
        assert ttr == 0.5454545454545454
        ttr, ci = document.complexity("ttr", window=3)
        assert ttr == 0.5555555555555555
        assert ci == 0.17781481095759366
        orlov_z = document.complexity("orlov_z", max_iterations=1)
        assert orlov_z == 7.461820552205992
        orlov_z = document.complexity("orlov_z", window=3, max_iterations=1)
        assert orlov_z == 0.3333333333333333
        cttr = document.complexity("cttr")
        assert cttr == 1.2792042981336627
        cttr = document.complexity("cttr", window=3)
        assert cttr == 0.6804138174397717


class TestCorpus:
    def test_sparse_error(self, document):
        with pytest.raises(NotImplementedError):
            cophi.model.Corpus([document], sparse=True)

    def test_dtm(self, corpus):
        assert corpus.dtm.sum().sum() == len(TOKENS)

    def test_map_metadata(self, corpus):
        metadata = pd.DataFrame({"uuid": "document", "A": "metadata"}, index=[1])
        matrix = corpus.map_metadata(corpus.dtm, metadata, fields=["A"])
        assert matrix.sum().sum() == len(TOKENS)
        assert "metadata" in matrix.index
        assert "document" not in matrix.index

    def test_stats(self, corpus):
        assert corpus.stats.sum() == 21

    def test_freq_spectrum(self, corpus):
        assert corpus.freq_spectrum.sum() == len(set(TOKENS))

    def test_types(self, corpus):
        assert len(corpus.types) == len(set(TOKENS))

    def test_sort(self, corpus):
        assert corpus.sort(corpus.dtm).sum().sum() == len(TOKENS)

    def test_mfw(self, corpus):
        assert corpus.mfw(1, as_list=False, rel=False).sum() == 3
        assert round(corpus.mfw(1, rel=True, as_list=False).sum()) == 0
        assert len(corpus.mfw(1, as_list=True, rel=False)) == 1

    def test_drop(self, corpus):
        matrix = corpus.drop(corpus.dtm, ["a"])
        assert "a" not in matrix.columns

    def test_cull(self, corpus):
        assert corpus.cull(corpus.dtm, 1).sum().sum() == len(TOKENS)

    def test_zscores(self, corpus):
        assert corpus.zscores.sum().sum() == 0

    def test_rel(self, corpus):
        assert round(corpus.rel.sum().sum()) == 1

    def test_tfidf(self, corpus):
        assert corpus.tfidf.sum().sum() == 0

    def test_num_types(self, corpus):
        assert corpus.num_types.sum().sum() == len(set(TOKENS))

    def test_num_tokens(self, corpus):
        assert corpus.num_tokens.sum().sum() == len(TOKENS)

    def test_complexity(self, corpus):
        ttr = corpus.complexity(window=3, measure="ttr")
        assert ttr.sum().sum() == 0.7333703665131491
        cttr = corpus.complexity(window=3, measure="cttr")
        assert cttr.sum().sum() == 0.6804138174397717

    def test_ttr(self, corpus):
        assert corpus.ttr == 0.5454545454545454

    def test_guiraud_r(self, corpus):
        assert corpus.guiraud_r == 1.8090680674665818

    def test_herdan_c(self, corpus):
        assert corpus.herdan_c == 0.7472217363092141

    def test_dugast_k(self, corpus):
        assert corpus.dugast_k == 2.0486818235486686

    def test_dugast_u(self, corpus):
        assert corpus.dugast_u == 2.0486818235486686

    def test_maas_a2(self, corpus):
        assert corpus.maas_a2 == 0.1054167238070372

    def test_tuldava_ln(self, corpus):
        assert corpus.tuldava_ln == -0.40544815832912834

    def test_brunet_w(self, corpus):
        assert corpus.brunet_w == 26.138632903383154

    def test_cttr(self, corpus):
        assert corpus.cttr == 1.2792042981336627

    def test_summer_s(self, corpus):
        assert corpus.summer_s == 0.6668234928556862

    def test_sichel_s(self, corpus):
        assert corpus.sichel_s == 0.16666666666666666

    def test_michea_m(self, corpus):
        assert corpus.michea_m == 6

    def test_honore_h(self, corpus):
        assert corpus.honore_h == 479.57905455967415

    def test_entropy(self, corpus):
        assert corpus.entropy == 1.6726254461503205

    def test_yule_k(self, corpus):
        assert corpus.yule_k == -661.1570247933887

    def test_simpson_d(self, corpus):
        assert corpus.simpson_d == 0.08181818181818183

    def test_herdan_vm(self, corpus):
        assert corpus.herdan_vm == 0.1998622114889836

    def test_orlov_z(self, corpus):
        assert corpus.orlov_z(max_iterations=1) == 7.461820552205992

    def test_svmlight(self, corpus):
        output = pathlib.Path("corpus.svmlight")
        corpus.svmlight(corpus.dtm, output)
        assert output.exists()
        with output.open("r", encoding="utf-8") as file:
            assert file.read() == "document document a:3 b:2 c:3 d:1 e:1 f:1\n"

    def test_plaintext(self, corpus):
        output = pathlib.Path("corpus.txt")
        corpus.plaintext(corpus.dtm, output)
        assert output.exists()
        with output.open("r", encoding="utf-8") as file:
            assert file.read() == "document document a a a b b c c c d e f\n"
