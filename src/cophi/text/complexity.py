"""
cophi.text.complexity
~~~~~~~~~~~~~~~~~~~~~

This module implements measures that assess the linguistic
and stylistic complexity of (literary) texts.

:math:`N` is the Absolute number of tokens, and :math:`V` the
Absolute number of types. :math:`H` is the Absolute number of
types occuring only once (hapax legomena), :math:`D` is the
absolute number of types occuring twice (dislegomena).

The code module was taken from
`here <https://github.com/tsproisl/Linguistic_and_Stylistic_Complexity>`_.
"""

import math
import numpy as np


# use num_types + num_tokens (int):

def ttr(num_types, num_tokens):
    """Calculate Type-Token Ratio (TTR).

    Used formula:
        .. math::
            TTR = \frac{V}{N}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return num_types / num_tokens


def guiraud_r(num_types, num_tokens):
    """Calculate Guiraud’s R (1954).

    Used formula:
        .. math::
            R = \frac{V}{\sqrt{N}}

    Parameters:
            num_types (int): Absolute number of types.
            num_tokens (int): Absolute number of tokens.
    """
    return num_types / math.sqrt(num_tokens)


def herdan_c(num_types, num_tokens):
    """Calculate Herdan’s C (1960, 1964).

    Used formula:
        .. math::
            C = \frac{\log{V}}{\log{N}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return math.log(num_types) / math.log(num_tokens)


def dugast_k(num_types, num_tokens):
    """Calculate Dugast’s k (1979).

    Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return math.log(num_types) / math.log(math.log(num_tokens))


def maas_a2(num_types, num_tokens):
    """Calculate Maas’ a^2 (1972).

    Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return (math.log(num_tokens)
            - math.log(num_types)) / (math.log(num_tokens) ** 2)


def dugast_u(num_types, num_tokens):
    """Calculate Dugast’s U (1978, 1979).

    Used formula:
        .. math::
            U = \frac{\log{N^2}}{\log{N} \; - \; \log{V}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return (math.log(num_tokens) ** 2) / (math.log(num_tokens)
                                          - math.log(num_types))


def tuldava_ln(num_types, num_tokens):
    """Calculate Tuldava’s LN (1977).

    Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return (1 - (num_types ** 2)) / ((num_types ** 2) * math.log(num_tokens))


def brunet_w(num_types, num_tokens):
    """Calculate Brunet’s W (1978).

    Used formula:
        .. math::
            W = V^{V^{0.172}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    a = -0.172
    return num_tokens ** (num_types ** -a)


def cttr(num_types, num_tokens):
    """Calculate Carroll’s Corrected Type-Token Ration (CTTR) (1964).

    Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return num_types / math.sqrt(2 * num_tokens)


def summer_s(num_types, num_tokens):
    """Calculate Summer’s S.

    Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
    """
    return math.log(math.log(num_types)) / math.log(math.log(num_tokens))


# use num_types + part of freq_spectrum:

def sichel_s(num_types, freq_spectrum):
    """Calculate Sichel’s S (1975).

    Used formula:
        .. math::
            S = \frac{D}{V}

    Parameters:
        num_types (int): Absolute number of types.
        freq_spectrum (dict): Counted occurring frequencies.
    """
    return freq_spectrum[2] / num_types


def michea_m(num_types, freq_spectrum):
    """Calculate Michéa’s M (1969, 1971).

    Used formula:
        .. math::
            M = \frac{V}{D}

    Parameters:
        num_types (int): Absolute number of types.
        freq_spectrum (dict): Counted occurring frequencies.
    """
    return num_types / freq_spectrum[2]


def honore_h(num_types, num_tokens, freq_spectrum):
    """Calculate Honoré’s H (1979).

    Used formula:
        .. math::
            H = 100 \cdot \frac{\log{N}}{1 - \frac{H}{V}}

    Parameters:
        num_types (int): Absolute number of types.
        freq_spectrum (dict): Counted occurring frequencies.
    """
    return 100 * (math.log(num_tokens)
                  / (1 - ((freq_spectrum[1]) / (num_types))))


# use num_tokens + freq_spectrum:

def entropy(num_tokens, freq_spectrum):
    """Calculate entropy S.

    Parameters:
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (pd.Series): Counted occurring frequencies.
    """
    a = freq_spectrum.index.values / num_tokens
    b = - np.log(a)
    result = freq_spectrum.values * a * b
    return result.sum()


def yule_k(num_tokens, freq_spectrum):
    """Calculate Yule’s K (1944).

    Used formula:
        .. math::
            K = 10^4 \times \frac{(\sum_{X=1}^{X}{{f_X}X^2}) - N}{N^2}

    Parameters:
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (pd.Series): Counted occurring frequencies.
    """
    a = freq_spectrum.index.values / num_tokens
    b = 1 / num_tokens
    return 10 ** 4 * ((freq_spectrum.values * a ** 2) - b).sum()


def simpson_d(num_tokens, freq_spectrum):
    """Calculate Simpson’s D.

    Parameters:
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (pd.Series): Counted occurring frequencies.
    """
    a = freq_spectrum.values / num_tokens
    b = freq_spectrum.index.values - 1
    return (freq_spectrum.values * a * (b / (num_tokens - 1))).sum()


def herdan_vm(num_types, num_tokens, freq_spectrum):
    """Calculate Herdan’s VM (1955).

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (pd.Series): Counted occurring frequencies.
    """
    a = freq_spectrum.index.values / num_tokens
    b = 1 / num_types
    return math.sqrt((freq_spectrum * a ** 2).sum() - b)


# use probabilistic models:

def orlov_z(num_tokens, num_types, freq_spectrum,
            max_iterations=100, min_tolerance=1):
    """Calculate Orlov’s Z (1983), approximated via Newton’s method.

    Parameters:
        num_types (int): Absolute number of types.
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (dict): Counted occurring frequencies.
    """
    most_frequent = max(freq_spectrum)
    p_star = most_frequent / num_tokens
    z = num_tokens / 2
    for i in range(max_iterations):
        next_z = z - (_get_z(num_tokens,
                             num_types,
                             p_star, z) / _derivative(num_tokens,
                                                      num_types,
                                                      p_star,
                                                      z))
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    return z


def _get_z(num_tokens, num_types, p_star, z):
    """Private function for :func:`orlov_z`.
    """
    return (((z / math.log(p_star * z))
            * (num_tokens / (num_tokens - z))
            * math.log(num_tokens / z) - num_types))


def _derivative(num_tokens, num_types, p_star, z):
    """Private function for :func:`orlov_z`.
    """
    return ((num_tokens
             * ((z - num_tokens)
             * math.log(p_star * z)
             + math.log(num_tokens / z)
             * (num_tokens * math.log(p_star * z) - num_tokens + z)))
            / (((num_tokens - z) ** 2) * (math.log(p_star * z) ** 2)))


# other:

def ci(results):
    """Calculate  the confidence interval for standardized TTR.

    Parameters:
        results (list): Bootstrapped TTRs.
    """
    return 1.96 * np.std(results) / math.sqrt(len(results))


def wrapper(measure):
    """A wrapper for all complexity functions.
    """
    measures = {"ttr": ttr,
                "guiraud_r": guiraud_r,
                "herdan_c": herdan_c,
                "dugast_k": dugast_k,
                "maas_a2": maas_a2,
                "dugast_u": dugast_u,
                "tuldava_ln": tuldava_ln,
                "brunet_w": brunet_w,
                "cttr": cttr,
                "summer_s": summer_s,
                "sichel_s": sichel_s,
                "michea_m": michea_m,
                "honore_h": honore_h,
                "entropy": entropy,
                "yule_k": yule_k,
                "simpson_d": simpson_d,
                "herdan_vm": herdan_vm,
                "orlov_z": orlov_z}
    return measures[measure]
