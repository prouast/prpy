[![Tests](https://github.com/prouast/propy/actions/workflows/main.yml/badge.svg)](https://github.com/prouast/propy/actions/workflows/main.yml)

# propy

A collection of Python utilities for signal, image, and video processing with `numpy`, `ffmpeg`, `tensorflow`, and `torch`.

## Dependencies

General requirements are `python>=3.8` and `ffmpeg`.

To install further required Python packages:

```
pip install -r requirements.txt
```

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