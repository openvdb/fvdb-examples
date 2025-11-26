#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Setup script for point_transformer_v3 project.

This installs the package in editable mode and ensures that:
- fvdb_extensions is importable
- pointcept (from external/pointcept) is importable

Install with: pip install -e .
"""

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent.resolve()

# Read requirements
requirements = []
requirements_path = PROJECT_ROOT / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("--"):
                requirements.append(line)

setup(
    name="point-transformer-v3",
    version="0.1.0",
    description="Point Transformer V3 with FVDB extensions",
    python_requires=">=3.8",
    packages=find_packages(where="external/pointcept")
    + find_packages(where=".", include=["fvdb_extensions", "fvdb_extensions.*"]),
    package_dir={
        "": "external/pointcept",
        "fvdb_extensions": "fvdb_extensions",
    },
    install_requires=requirements,
)
