"""
cophi_toolbox.complexity
~~~~~~~~~~~~~~~~~~~~~~~~

This module provides 
"""

import numpy as np


# types + tokens:

def ttr(tokens, types):
    """"""
    return types / tokens


def guiraud_r(tokens, types):
    """Guiraud (1954)"""
    return types / np.sqrt(tokens)


def herdan_c(tokens, types):
    """Herdan (1960, 1964)"""
    return np.log(types) / np.log(tokens)


def dugast_k(tokens, types):
    """Dugast (1979)"""
    return np.log(types) / np.log(np.log(tokens))


def maas_a2(tokens, types):
    """Maas (1972)"""
    return (np.log(tokens) - np.log(types)) / (np.log(tokens) ** 2)


def dugast_u(tokens, types):
    """Dugast (1978, 1979)"""
    return (np.log(tokens) ** 2) / (np.log(tokens) - np.log(types))


def tuldava_ln(tokens, types):
    """Tuldava (1977)"""
    return (1 - (types ** 2)) / ((types ** 2) * np.log(tokens))


def brunet_w(tokens, types):
    """Brunet (1978)"""
    a = -0.172
    return tokens ** (types ** -a)


def cttr(tokens, types):
    """Carroll's Corrected Type-Token Ration"""
    return types / np.sqrt(2 * tokens)


def summer_s(tokens, types):
    """Summer's S index"""
    return np.log(np.log(types)) / np.log(np.log(tokens))


# types + part of the frequency spectrum:

def sichel_s(types, freq_spectrum):
    """Sichel (1975)"""
    return freq_spectrum[2] / types


def michea_m(types, freq_spectrum):
    """Michéa (1969, 1971)"""
    return types / freq_spectrum[2]


def honore_h(tokens, types, freq_spectrum):
    """Honoré (1979)"""
    return 100 * (np.log(tokens) / (1 - ((freq_spectrum[1]) / (types))))


# tokens + frequency spectrum:

def entropy(tokens, freq_spectrum):
    """"""
    a = -np.log(freq_spectrum.index / tokens)
    b = freq_spectrum / tokens
    return (freq_spectrum * a * b).sum()

def yule_k(tokens, freq_spectrum):
    """Yule (1944)"""
    a = freq_spectrum.index / tokens
    b = 1 / tokens
    return 10 ** 4 * ((freq_spectrum * a ** 2) - b).sum()

def simpson_d(tokens, freq_spectrum):
    """"""
    a = freq_spectrum / tokens
    b = freq_spectrum.index - 1
    return (freq_spectrum * a * (b / (tokens - 1))).sum()

def herdan_vm(tokens, types, freq_spectrum):
    """Herdan (1955)"""
    a = freq_spectrum / tokens
    b = 1 / types
    return np.sqrt(((freq_spectrum * a ** 2) - b).sum())

'''
def hdd(tokens, freq_spectrum, sample_size=42):
    """McCarthy and Jarvis (2010)"""
    return sum(((1 - scipy.stats.hypergeom.pmf(0, tokens, freq, sample_size)) / sample_size for word, freq in freq_spectrum.items()))
'''

# probabilistic models:

def orlov_z(tokens, types, freq_spectrum, max_iterations=100, min_tolerance=1):
    """Orlov (1983)

    Approximation via Newton's method.

    """
    def function(tokens, types, p_star, z):
        return (z / np.log(p_star * z)) * (tokens / (tokens - z)) * np.log(tokens / z) - types

    def derivative(tokens, types, p_star, z):
        """Derivative obtained from WolframAlpha:
        https://www.wolframalpha.com/input/?x=0&y=0&i=(x+%2F+(log(p+*+x)))+*+(n+%2F+(n+-+x))+*+log(n+%2F+x)+-+v

        """
        return (tokens * ((z - tokens) * np.log(p_star * z) + np.log(tokens / z) * (tokens * np.log(p_star * z) - tokens + z))) / (((tokens - z) ** 2) * (np.log(p_star * z) ** 2))
    most_frequent = freq_spectrum.max()
    p_star = most_frequent / tokens
    z = tokens / 2
    for i in range(max_iterations):
        next_z = z - (function(tokens, types, p_star, z) / derivative(tokens, types, p_star, z))
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    else:
        print("Exceeded max_iterations")
    return z

def ci(results):
    """calculate the confidence interval for sttr """
    return 1.96 * np.std(results) / np.sqrt(len(results))