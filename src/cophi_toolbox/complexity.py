"""
cophi_toolbox.complexity
~~~~~~~~~~~~~~~~~~~~~~~~

This module provides measures that assess the linguistic 
and stylistic complexity of (literary) texts.
"""

from typing import Union, Dict, List
import numpy as np


# use sum_types + sum_tokens:

def ttr(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Type-Token Ratio.

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return sum_types / sum_tokens

def guiraud_r(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Guiraud’s R (1954)

    Parameters:
            sum_types: Number of types.
            sum_tokens: Number of tokens.
        """
    return sum_types / np.sqrt(sum_tokens)

def herdan_c(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Herdan’s C (1960, 1964).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return np.log(sum_types) / np.log(sum_tokens)

def dugast_k(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Dugast’s k (1979).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return np.log(sum_types) / np.log(np.log(sum_tokens))

def maas_a2(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Maas’ a^2 (1972).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return (np.log(sum_tokens) - np.log(sum_types)) / (np.log(sum_tokens) ** 2)

def dugast_u(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Dugast’s U (1978, 1979).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return (np.log(sum_tokens) ** 2) / (np.log(sum_tokens) - np.log(sum_types))

def tuldava_ln(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Tuldava’s LN (1977).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return (1 - (sum_types ** 2)) / ((sum_types ** 2) * np.log(sum_tokens))

def brunet_w(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Brunet’s W (1978)

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    a = -0.172
    return sum_tokens ** (sum_types ** -a)

def cttr(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Carroll’s Corrected Type-Token Ration (1964).

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return sum_types / np.sqrt(2 * sum_tokens)

def summer_s(sum_types: int, sum_tokens: int) -> Union[int, float]:
    """Calculate Summer’s S.

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
    """
    return np.log(np.log(sum_types)) / np.log(np.log(sum_tokens))


# use sum_types + part of the frequency spectrum:

def sichel_s(sum_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Sichel’s S (1975).

    Parameters:
        sum_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return freq_spectrum[2] / sum_types

def michea_m(sum_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Michéa’s M (1969, 1971).

    Parameters:
        sum_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return sum_types / freq_spectrum[2]

def honore_h(sum_tokens: int, sum_types: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Honoré’s H (1979).

    Parameters:
        sum_types: Number of types.
        freq_spectrum: Counted occurring frequencies.
    """
    return 100 * (np.log(sum_tokens) / (1 - ((freq_spectrum[1]) / (sum_types))))


# use sum_tokens + frequency spectrum:

def entropy(sum_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate entropy.

    Parameters:
        sum_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = -np.log(freq_spectrum.index / sum_tokens)
    b = freq_spectrum / sum_tokens
    return (freq_spectrum * a * b).sum()

def yule_k(sum_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Yule’s K (1944).

    Parameters:
        sum_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum.index / sum_tokens
    b = 1 / sum_tokens
    return 10 ** 4 * ((freq_spectrum * a ** 2) - b).sum()

def simpson_d(sum_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Simpson’s D.

    Parameters:
        sum_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum / sum_tokens
    b = freq_spectrum.index - 1
    return (freq_spectrum * a * (b / (sum_tokens - 1))).sum()

def herdan_vm(sum_types: int, sum_tokens: int, freq_spectrum: Dict[int, int]) -> Union[int, float]:
    """Calculate Herdan (1955)

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    a = freq_spectrum / sum_tokens
    b = 1 / sum_types
    return np.sqrt(((freq_spectrum * a ** 2) - b).sum())


# use probabilistic models:

def orlov_z(sum_tokens: int, sum_types: int, freq_spectrum: Dict[int, int],
            max_iterations: int = 100, min_tolerance: int = 1) -> Union[int, float]:
    """Calculate Orlov’s Z (1983), approximated via Newton’s method.

    Parameters:
        sum_types: Number of types.
        sum_tokens: Number of tokens.
        freq_spectrum: Counted occurring frequencies.
    """
    def function(sum_tokens: int, sum_types: int, p_star, z) -> Union[int, float]:
        return (z / np.log(p_star * z)) * (sum_tokens / (sum_tokens - z)) * np.log(sum_tokens / z) - sum_types

    def derivative(sum_tokens: int, sum_types: int, p_star, z) -> Union[int, float]:
        return (sum_tokens * ((z - sum_tokens) * np.log(p_star * z) + np.log(sum_tokens / z) * (sum_tokens * np.log(p_star * z) - sum_tokens + z))) / (((sum_tokens - z) ** 2) * (np.log(p_star * z) ** 2))
    most_frequent = freq_spectrum.max()
    p_star = most_frequent / sum_tokens
    z = sum_tokens / 2
    for i in range(max_iterations):
        next_z = z - (function(sum_tokens, sum_types, p_star, z) / derivative(sum_tokens, sum_types, p_star, z))
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