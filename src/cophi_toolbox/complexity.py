"""
cophi_toolbox.complexity
~~~~~~~~~~~~~~~~~~~~~~~~

This module provides measures that assess the linguistic and stylistic 
complexity of (literary) texts.

Regarding the formulas, :math:`N` is the number of tokens, and :math:`V` 
the number of types. :math:`H` is the number of types occuring only once 
(hapax legomena), :math:`D` is the number of types occuring twice (dislegomena).

This module was taken from `here <https://github.com/tsproisl/Linguistic_and_Stylistic_Complexity>`_.
"""

from typing import Union, Dict, List

import numpy as np


# use num_types + num_tokens:

def ttr(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Type-Token Ratio (TTR).

    Used formula:
        .. math::
            TTR = \frac{V}{N}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return num_types / num_tokens

def guiraud_r(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Guiraud’s R (1954).

    Used formula:
        .. math::
            R = \frac{V}{\sqrt{N}}

    Parameters:
            num_types: Number of types.
            num_tokens: Number of tokens.
        """
    return num_types / np.sqrt(num_tokens)

def herdan_c(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Herdan’s C (1960, 1964).

    Used formula:
        .. math::
            C = \frac{\log{V}}{\log{N}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return np.log(num_types) / np.log(num_tokens)

def dugast_k(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Dugast’s k (1979).

    Used formula:
        .. math::
            k = \frac{\log{V}}{\log{\log{N}}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return np.log(num_types) / np.log(np.log(num_tokens))

def maas_a2(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Maas’ a^2 (1972).

    Used formula:
        .. math::
            a^2 = \frac{\log{N} \; - \; \log{V}}{\log{N}^2}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return (np.log(num_tokens) - np.log(num_types)) / (np.log(num_tokens) ** 2)

def dugast_u(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Dugast’s U (1978, 1979).

    Used formula:
        .. math::
            U = \frac{\log{N^2}}{\log{N} \; - \; \log{V}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return (np.log(num_tokens) ** 2) / (np.log(num_tokens) - np.log(num_types))

def tuldava_ln(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Tuldava’s LN (1977).

    Used formula:
        .. math::
            LN = \frac{1 \; - \; V^2}{V^2 \; \cdot \; \log{N}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return (1 - (num_types ** 2)) / ((num_types ** 2) * np.log(num_tokens))

def brunet_w(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Brunet’s W (1978).

    Used formula:
        .. math::
            W = V^{V^{0.172}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    a = -0.172
    return num_tokens ** (num_types ** -a)

def cttr(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Carroll’s Corrected Type-Token Ration (CTTR) (1964).

    Used formula:
        .. math::
            CTTR = \frac{V}{\sqrt{2 \; \cdot \; N}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return num_types / np.sqrt(2 * num_tokens)

def summer_s(num_types: int, num_tokens: int) -> Union[int, float]:
    """Calculate Summer’s S.

    Used formula:
        .. math::
            S = \frac{\log{\log{V}}}{\log{\log{N}}}

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
    """
    return np.log(np.log(num_types)) / np.log(np.log(num_tokens))


# use num_types + part of freq_spectrum:

def sichel_s(num_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Sichel’s S (1975).

    Used formula:
        .. math::
            S = \frac{D}{V}

    Parameters:
        num_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return freq_spectrum[2] / num_types

def michea_m(num_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Michéa’s M (1969, 1971).

    Used formula:
        .. math::
            M = \frac{V}{D}

    Parameters:
        num_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return num_types / freq_spectrum[2]

def honore_h(num_tokens: int, num_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Honoré’s H (1979).

    Used formula:
        .. math::
            H = 100 \cdot \frac{\log{N}}{1 - \frac{H}{V}}

    Parameters:
        num_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return 100 * (np.log(num_tokens) / (1 - ((freq_spectrum[1]) / (num_types))))


# use num_tokens + freq_spectrum:

def entropy(num_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate entropy S.

    Parameters:
        num_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = -np.log(freq_spectrum.index / num_tokens)
    b = freq_spectrum / num_tokens
    return (freq_spectrum * a * b).sum()

def yule_k(num_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Yule’s K (1944).

    Used formula:
        .. math::
            K = 10^4 \times \frac{(\sum_{X=1}^{X}{{f_X}X^2}) - N}{N^2}

    Parameters:
        num_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum.index / num_tokens
    b = 1 / num_tokens
    return 10 ** 4 * ((freq_spectrum * a ** 2) - b).sum()

def simpson_d(num_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Simpson’s D.

    Parameters:
        num_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum / num_tokens
    b = freq_spectrum.index - 1
    return (freq_spectrum * a * (b / (num_tokens - 1))).sum()

def herdan_vm(num_types: int, num_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Herdan’s VM (1955).

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum / num_tokens
    b = 1 / num_types
    return np.sqrt(((freq_spectrum * a ** 2) - b).sum())


# use probabilistic models:

def orlov_z(num_tokens: int, num_types: int, freq_spectrum: Dict[int, int],
            max_iterations: int = 100, min_tolerance: int = 1) -> Union[int, float]:
    """Calculate Orlov’s Z (1983), approximated via Newton’s method.

    Parameters:
        num_types: Number of types.
        num_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    def function(num_tokens, num_types, p_star, z):
        return (z / np.log(p_star * z)) * (num_tokens / (num_tokens - z)) * np.log(num_tokens / z) - num_types

    def derivative(num_tokens, num_types, p_star, z):
        return (num_tokens * ((z - num_tokens) * np.log(p_star * z) + np.log(num_tokens / z) * (num_tokens * np.log(p_star * z) - num_tokens + z))) / (((num_tokens - z) ** 2) * (np.log(p_star * z) ** 2))
    most_frequent = freq_spectrum.max()
    p_star = most_frequent / num_tokens
    z = num_tokens / 2
    for i in range(max_iterations):
        next_z = z - (function(num_tokens, num_types, p_star, z) / derivative(num_tokens, num_types, p_star, z))
        abs_diff = abs(z - next_z)
        z = next_z
        if abs_diff <= min_tolerance:
            break
    return z


# other:

def ci(results: List[Union[int, float]]) -> Union[int, float]:
    """Calculate  the confidence interval for standardized TTR.

    Parameters:
        results: Bootstrapped TTRs.
    """
    return 1.96 * np.std(results) / np.sqrt(len(results))