[![Tests](https://github.com/prouast/propy/actions/workflows/main.yml/badge.svg)](https://github.com/prouast/propy/actions/workflows/main.yml)

# propy

A collection of Python utilities for signal, image, and video processing.
It contains subpackages for working with `numpy`, `ffmpeg`, `tensorflow`, and `torch`.

## Installation

General prerequisites are `python>=3.8` and `ffmpeg` installed and accessible via the `$PATH` environment variable.

To install `propy` and its dependencies:

```
git clone https://github.com/prouast/propy.git
pip install "./propy[ffmpeg,numpy,tensorflow,torch,test]"
```

The above runs a full install of all dependencies.
It is possible to customize the install of the dependencies by only listing the desired subpackages out of `ffmpeg`, `numpy`, `tensorflow`, `torch`, and `test` in the square brackets above.

## Linting and tests

To lint and run tests:

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest
```

## Build

To build:

```
python -m build
```
