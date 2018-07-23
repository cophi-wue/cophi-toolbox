import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def corpus_model():
    a = pd.Series(["A", "B", "C", "D", "E"], name="a")
    b = pd.Series(["A", "B", "F", "G", "E"], name="b")
    c = pd.Series(["H", "I", "C", "J", "E"], name="c")
    d = pd.Series(["A", "K", "L", "D", "M"], name="d")
    e = pd.Series(["N", "O", "B", "A", "E"], name="e")
    documents = [a, b, c, d, e]
    return ct.corpus(documents)

class TestCorpus:
    def test_size(corpus_model):
        assert corpus_model.size["documents"] == 5
        assert corpus_model.size["types"] == 15

    def test_dtm(corpus_model):
        assert list(corpus_model.columns) == "ABCDEFGHIKLMNO".split()
        assert list(corpus_model.index) == "abcde".split()
        assert corpus_model.sum().sum() == 25.0

    def test_sorted_dtm(corpus_model):
        assert list(corpus_model.sorted_dtm.columns) == "BAECDFGHIJKLMNOP".split()

    def test_vocabulary(corpus_model):
        assert corpus_model.vocabulary == "ABCDEFGHIKLMNO".split()

    def test_freq_spectrum(corpus_model):
        assert corpus_model.freq_spectrum.sum() == 15
    
    def test_get_mfw(corpus_model):
        assert corpus_model.get_mfw(n=1) == ["A"]
    
    def test_get_hl(corpus_model):
        assert corpus_model.get_hl() == "BCDEFGHIKLMNO".split()

    def test_drop(corpus_model):
        assert corpus_model.drop(corpus_model.dtm, ["A"]).sum().sum() == 20.0

    def test_zscores(corpus_model):
        assert corpus_model.zscores.sum().sum() == -1.6653345369377348e-16

    def test_rel_freqs(corpus_model):
        assert corpus_model.rel_freqs.sum().sum() == 5.0

    def test_tfidf(corpus_model):
        assert corpus_model.tfidf.sum().sum() == 5.169951211177811

    def test_sum_tokens(corpus_model):
        assert corpus_model.sum_tokens.sum() == 25.0

    def test_sum_types(corpus_model):
        assert corpus_model.sum_types.sum() == 24.0

    def test_get_ttr(corpus_model):
        assert corpus_model.get_ttr().sum() == 4.8

    def test_ttr(corpus_model):
        assert corpus_model.ttr == 0.96

    def test_guiraud_r(corpus_model):
        assert corpus_model.guiraud_r == 4.8

    def test_get_guiraud_r(corpus_model):
        assert corpus_model.get_guiraud_r().sum() == 10.73312629199899

    def test_herdan_c(corpus_model):
        assert corpus_model.herdan_c == 0.9873179343530823

    def test_get_herdan_c(corpus_model):
        assert corpus_model.get_herdan_c().sum() == 4.861353116146786

    def test_dugast_k(corpus_model):
        assert corpus_model.dugast_k == 2.718534096750976

    def test_get_dugast_k(corpus_model):
        assert corpus_model.get_dugast_k().sum() == 16.441043713677615

    def test_maas_a2(corpus_model):
        assert corpus_model.maas_a2 == 0.003939905214404149

    def test_get_maas_a2(corpus_model):
        assert corpus_model.get_maas_a2().sum() == 0.08614615250583076

    def test_tuldava_ln(corpus_model):
        assert corpus_model.tuldava_ln == -0.3101281140380007

    def test_get_tuldava_ln(corpus_model):
        assert corpus_model.get_tuldava_ln().sum() == -2.968427649858546

    def test_brunet_w(corpus_model):
        assert corpus_model.brunet_w == 259.9085568722845

    def test_get_brunet_w(corpus_model):
        assert corpus_model.get_brunet_w().sum() == 41.12829855223466

    def test_cttr(corpus_model):
        assert corpus_model.cttr == 3.394112549695428

    def test_get_cttr(corpus_model):
        assert corpus_model.get_cttr().sum() == 7.58946638440411

    def test_summer_s(corpus_model):
        assert corpus_model.summer_s == 0.9890822769947938

    def test_get_summer_s(corpus_model):
        assert corpus_model.get_summer_s().sum() == 4.686372260494915

    def test_entropy(corpus_model):
        assert corpus_model.entropy == 16.63780936225843

    def test_yule_k(corpus_model):
        assert corpus_model.yule_k == -688.0

    def test_simpson_d(corpus_model):
        assert corpus_model.simpson_d == 0.025

    def test_herdan_vm(corpus_model):
        assert corpus_model.herdan_vm == 1.417509553171806