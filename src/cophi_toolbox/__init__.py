"""
cophi_toolbox
~~~~~~~~~~~~~

This is an NLP preprocessing library for handling, modeling and processing text data. You
can easily pipe a collection of text files using the high-level API:

```
corpus, metadata = cophi_toolbox.pipe(directory="british-fiction-corpus",
                                      pathname_pattern="**/*.txt",
                                      encoding="utf-8",
                                      lowercase=True,
                                      ngrams=1,
                                      token_pattern=r"\p{L}+\p{P}?\p{L}+")
```

There are also a plenty of complexity metrics for measuring lexical richness of (literary) texts.

Measures that use sample size and vocabulary size:
  * Type-Token Ratio (:func:`ttr`).
  * Guiraud’s :math:`R` (:func:`guiraud_r`).
  * Herdan’s :math:`C` (:func:`herdan_c`).
  * Dugast’s :math:`k` (:func:`dugast_k`).
  * Maas’ :math:`a^2` (:func:`maas_a2`).
  * Dugast’s :math:`U` (:func:`dugast_u`).
  * Tuldava’s :math:`LN` (:func:`tuldava_ln`).
  * Brunet’s :math:`W` (:func:`brunet_w`).
  * Carroll’s :math:`CTTR` (:func:`cttr`).
  * Summer’s :math:`S` (:func:`summer_s`).

Measures that use part of the frequency spectrum:
  * Honoré’s :math:`H` (:func:`honore_h`).
  * Sichel’s :math:`S` (:func:`sichel_s`).
  * Michéa’s :math:`M` (:func:`michea_m`).

Measures that use the whole frequency spectrum:
  * Entropy :math:`S` (:func:`entropy`).
  * Yule’s :math:`K` (:func:`yule_k`).
  * Simpson’s :math:`D` (:func:`simpson_d`).
  * Herdan’s :math:`V_m` (:func:`herdan_vm`).

Parameters of probabilistic models:
  * Orlov’s :math:`Z` (:func:`orlov_z`).
"""

import logging

from .api import *
from .model import *
from .complexity import *
from .utils import *

logging.getLogger("cophi_toolbox").addHandler(logging.NullHandler())
