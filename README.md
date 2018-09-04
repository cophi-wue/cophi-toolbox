# A library for preprocessing
`cophi` is a Python library for handling, modeling and processing text corpora. You
can easily pipe a collection of text files using the high-level API:

```python
corpus, metadata = cophi.corpus(directory="british-fiction-corpus",
                                filepath_pattern="**/*.txt",
                                encoding="utf-8",
                                lowercase=True,
                                token_pattern=r"\p{L}+\p{P}?\p{L}+")
```

## Getting started
To install the latest **stable** version:
```
$ pip install cophi
```

To install the latest **development** version:
```
$ pip install --upgrade git+https://github.com/cophi-wue/cophi-toolbox.git@testing
```

Check out the introducing [Jupyter notebook](https://github.com/cophi-wue/cophi-toolbox/blob/master/notebooks/API.ipynb).

## Contents
- [`api`](https://github.com/cophi-wue/cophi-toolbox/blob/master/src/cophi_toolbox/api.py): High-level API.
- [`model`](https://github.com/cophi-wue/cophi-toolbox/blob/master/src/cophi_toolbox/model.py): Low-level model classes.
- [`complexity`](https://github.com/cophi-wue/cophi-toolbox/blob/master/src/cophi_toolbox/complexity.py): Measures that assess the linguistic and stylistic complexity of (literary) texts.
- [`utils`](https://github.com/cophi-wue/cophi-toolbox/blob/master/src/cophi_toolbox/utils.py): Low-level helper functions.


## Available complexity measures
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