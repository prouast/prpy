
# prpy

[![Tests](https://github.com/prouast/prpy/actions/workflows/main.yml/badge.svg)](https://github.com/prouast/prpy/actions/workflows/main.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/prpy?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20downloads)](https://pypi.org/project/prpy/)

A collection of Python utilities for signal, image, and video processing.
It contains subpackages for working with `numpy`, `ffmpeg`, `tensorflow`, and `torch`.

## Installation

General prerequisites are `python>=3.8` and `ffmpeg` installed and accessible via the `$PATH` environment variable.

- Please note: If using `numpy` or `tensorflow` options, we only support Python `<3.12` because of the dependencies. 

The easiest way to install the latest version of `prpy`:

```
pip install "prpy[ffmpeg,numpy,tensorflow,torch,test]"
```

Alternatively, it can be done by cloning the source:

```
git clone https://github.com/prouast/prpy.git
pip install "./prpy[ffmpeg,numpy,tensorflow,torch,test]"
```

The above run full installs of all dependencies.
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

## Re-build and re-install locally

```
pip uninstall -y prpy && pip install -e .
```
