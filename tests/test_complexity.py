import collections

import pandas as pd
import pytest

import cophi

P_STAR = 1
Z = 2
NUM_TYPES = 5
NUM_TOKENS = 8
MEASURES = {"ttr", "guiraud_r", "herdan_c", "dugast_k", "maas_a2", "dugast_u",
            "tuldava_ln", "brunet_w", "cttr", "summer_s", "sichel_s", "michea_m",
            "honore_h", "entropy", "yule_k", "simpson_d", "herdan_vm", "orlov_z"}


@pytest.fixture
def frequency_spectrum():
    tokens = ["A", "A", "A", "B", "B", "C", "D", "E"]
    freqs = collections.Counter(tokens)
    freq_spectrum = collections.Counter(freqs.values())
    return pd.Series(freq_spectrum)

def test_ttr():
    ttr = cophi.complexity.ttr(NUM_TYPES, NUM_TOKENS)
    assert ttr == 0.625

def test_guiraud_r():
    guiraud_r = cophi.complexity.guiraud_r(NUM_TYPES, NUM_TOKENS)
    assert guiraud_r == 1.7677669529663687

def test_herdan_c():
    herdan_c = cophi.complexity.herdan_c(NUM_TYPES, NUM_TOKENS)
    assert herdan_c == 0.7739760316291208

def test_dugast_k():
    dugast_k = cophi.complexity.dugast_k(NUM_TYPES, NUM_TOKENS)
    assert dugast_k == 2.198387244399397

def test_maas_a2():
    maas_a2 = cophi.complexity.maas_a2(NUM_TYPES, NUM_TOKENS)
    assert maas_a2 == 0.10869455276357046

def test_dugast_u():
    dugast_u = cophi.complexity.dugast_u(NUM_TYPES, NUM_TOKENS)
    assert dugast_u == 9.200093055032609

def test_tuldava_ln():
    tuldava_ln = cophi.complexity.tuldava_ln(NUM_TYPES, NUM_TOKENS)
    assert tuldava_ln == -0.4616624130844683

def test_brunet_w():
    brunet_w = cophi.complexity.brunet_w(NUM_TYPES, NUM_TOKENS)
    assert brunet_w == 15.527998381095463

def test_cttr():
    cttr = cophi.complexity.cttr(NUM_TYPES, NUM_TOKENS)
    assert cttr == 1.25

def test_summer_s():
    summer_s = cophi.complexity.summer_s(NUM_TYPES, NUM_TOKENS)
    assert summer_s == 0.650027873362293

def test_sichel_s(frequency_spectrum):
    sichel_s = cophi.complexity.sichel_s(NUM_TYPES, frequency_spectrum)
    assert sichel_s == 0.2

def test_michea_m(frequency_spectrum):
    michea_m = cophi.complexity.michea_m(NUM_TYPES, frequency_spectrum)
    assert michea_m == 5.0

def test_honore_h(frequency_spectrum):
    honore_h = cophi.complexity.honore_h(NUM_TYPES, NUM_TOKENS, frequency_spectrum)
    assert honore_h == 519.8603854199589

def test_entropy(frequency_spectrum):
    entropy = cophi.complexity.entropy(NUM_TOKENS, frequency_spectrum)
    assert entropy == 1.4941751382893083

def test_yule_k(frequency_spectrum):
    yule_k = cophi.complexity.yule_k(NUM_TOKENS, frequency_spectrum)
    assert yule_k == -1250.0

def test_simpson_d(frequency_spectrum):
    simpson_d = cophi.complexity.simpson_d(NUM_TOKENS, frequency_spectrum)
    assert simpson_d == 0.05357142857142857

def test_herdan_vm(frequency_spectrum):
    herdan_vm = cophi.complexity.herdan_vm(NUM_TYPES, NUM_TOKENS, frequency_spectrum)
    assert herdan_vm == 0.22360679774997894

def test_orlov_z(frequency_spectrum):
    orlov_z = cophi.complexity.orlov_z(NUM_TYPES, NUM_TOKENS, frequency_spectrum)
    assert orlov_z == 2.583892154363366

def test_get_z():
    z = cophi.complexity._get_z(NUM_TOKENS, NUM_TYPES, P_STAR, Z)
    assert z == 0.33333333333333304

def test_derivative():
    d = cophi.complexity._derivative(NUM_TOKENS, NUM_TYPES, P_STAR, Z)
    assert d == -2.2152246080002977

def test_ci():
    results = [1, 2, 3, 4, 5]
    ci = cophi.complexity.ci(results)
    assert ci == 1.2396128427860047

def test_wrapper():
    for measure in MEASURES:
        function = cophi.complexity.wrapper(measure)
        assert callable(function)
        assert function.__name__ == measure
