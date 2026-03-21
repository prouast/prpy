# Contributing & Development Guide

This guide covers setting up the development environment, running tests, and managing releases for **prpy**.

## Environment Setup

**prpy** contains a C extension (`image_ops.c`), requiring a C compiler on your system to build from source.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/prouast/prpy.git
    cd prpy
    ````
2. **Install with test dependencies:**
    Requires Python 3.10+.
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[test,ffmpeg,numpy,tensorflow,torch]"
    ```

## Testing and Linting

We use `pytest` for unit testing and `flake8` for linting.

  - **Run tests:**
    ```bash
    pytest
    ```
  - **Run linter:**
    ```bash
    flake8 . --count --select=F,E9 --show-source --statistics
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

## Release Workflow

Releases are fully automated via GitHub Actions using `cibuildwheel`. Pushing a version tag triggers the building of source distributions, `manylinux`/macOS/Windows wheels, PyPI publication, and GitHub Release creation.

1. **Tag the Release:** Ensure your local `main` branch is up to date and tag the commit. `setuptools_scm` will automatically derive the version from this tag.
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    ```
2. **Push:**
    ```bash
    git push origin main --follow-tags
    ```

Once pushed, the `.github/workflows/release.yml` workflow takes over.
