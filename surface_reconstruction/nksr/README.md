# Neural Kernel Surface Reconstruction (NKSR)

Idea: This is supposed to be a minimal implementation of NN-based surface reconstruction method.
Since the common belief is that NN-based methods are very scalable with more and more data, we should go with the very brute force network method instead of the more sophisticated ones (i.e. kernel solve).

## Installation

Prerequisites: Follow fVDB [official website](https://github.com/openvdb/fvdb-core) to install the conda environment of fVDB.

```bash
# Editable install
pip install --no-build-isolation -e .
```

## Usage

```bash
# Quick test
python examples/recons_simple.py
```
