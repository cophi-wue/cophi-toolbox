# A library for preprocessing

**cophi-toolbox** is a Python library for handling, modeling and processing text corpora. You
can easily pipe a collection of text files using the high-level API:

```
corpus, metadata = ct.pipe(directory="british-fiction-corpus",
                           pathname_pattern="**/*.txt",
                           encoding="utf-8",
                           lowercase=True,
                           ngrams=1,
                           token_pattern=r"\p{L}+\p{P}?\p{L}+")
```

There are also a plenty of complexity metrics for measuring lexical richness of (literary) texts.

Measures that use sample size and vocabulary size:
  * Type-Token Ratio TTR
  * Guiraud’s R
  * Herdan’s C
  * Dugast’s k
  * Maas’ a<sup>2</sup>
  * Dugast’s U
  * Tuldava’s LN
  * Brunet’s W
  * Carroll’s CTTR
  * Summer’s S

Measures that use part of the frequency spectrum:
  * Honoré’s H
  * Sichel’s S
  * Michéa’s M

Measures that use the whole frequency spectrum:
  * Entropy S
  * Yule’s K
  * Simpson’s D
  * Herdan’s V<sub>m</sub>

Parameters of probabilistic models:
  * Orlov’s Z