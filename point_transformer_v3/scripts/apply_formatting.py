#!/usr/bin/env python3

# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0


"""
Apply code formatting to point_transformer_v3 project.

This script applies black formatting to:
- scripts directory
- fvdb_extensions directory
- setup_env.py

It ignores the external directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()


def main():
    """Apply formatting using black."""
    # Directories and files to format
    targets = [
        str(PROJECT_ROOT / "scripts"),
        str(PROJECT_ROOT / "fvdb_extensions"),
        str(PROJECT_ROOT / "setup_env.py"),
    ]

    # Black options matching codestyle.yml
    black_options = [
        "--target-version=py311",
        "--line-length=120",
        "--verbose",
    ]

    # Run black via python -m for better portability
    cmd = [sys.executable, "-m", "black"] + black_options + targets

    print(f"Running: {' '.join(cmd)}")
    print(f"Formatting targets:")
    for target in targets:
        print(f"  - {target}")
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✓ Formatting applied successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Formatting failed with exit code {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("\n✗ Error: black not found. Please install it:")
        print("  pip install black~=24.0")
        return 1


if __name__ == "__main__":
    sys.exit(main())
