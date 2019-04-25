"""
This is a Python library for handling, modeling and processing text
corpora. You can easily pipe a collection of text files using the
high-level API:

```
corpus, metadata = cophi.corpus(directory="british-fiction-corpus",
                                pathname_pattern="**/*.txt",
                                encoding="utf-8",
                                lowercase=True,
                                token_pattern=r"\p{L}+\p{P}?\p{L}+")
```

There are also a plenty of complexity metrics for measuring the lexical
richness of (literary) texts.

Measures that use sample size and vocabulary size:
  * Type-Token Ratio :math:`TTR`
  * Guiraud’s :math:`R`
  * Herdan’s :math:`C`
  * Dugast’s :math:`k`
  * Maas’ :math:`a^2`
  * Dugast’s :math:`U`
  * Tuldava’s :math:`LN`
  * Brunet’s :math:`W`
  * Carroll’s :math:`CTTR`
  * Summer’s :math:`S`

Measures that use part of the frequency spectrum:
  * Honoré’s :math:`H`
  * Sichel’s :math:`S`
  * Michéa’s :math:`M`

Measures that use the whole frequency spectrum:
  * Entropy :math:`S`
  * Yule’s :math:`K`
  * Simpson’s :math:`D`
  * Herdan’s :math:`V_m`

Parameters of probabilistic models:
  * Orlov’s :math:`Z`

For a more detailed description and the used formulas, have a look at the
:module:`complexity` module.
"""


from cophi.api import document, corpus
