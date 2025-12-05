#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Setup script for fvdb-nksr project.

This installs the package in editable mode and ensures that:
- nksr and nksr_fvdb are importable

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
    name="fvdb-nksr",
    version="0.1.0",
    description="Surface reconstruction with FVDB using NKSR-inspired approach",
    python_requires=">=3.8",
    packages=find_packages(where=".", include=["nksr", "nksr.*"]),
    package_dir={"": "."},
    install_requires=requirements,
)
