#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Check for SPDX identifiers in source files.
Excludes external directory and hidden files.
"""

import os
import sys
from pathlib import Path

# Extensions to check
EXTENSIONS = {'.py', '.cpp', '.h', '.cu', '.cuh', '.sh'}

# Directories to exclude
EXCLUDES = {'external', '__pycache__', '.git', '.github', '.vscode', '.idea'}

def check_file(filepath):
    """Check if file contains SPDX-License-Identifier."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read first 20 lines
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                if "SPDX-License-Identifier" in line:
                    return True
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False
    return False

def main():
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent

    print(f"Checking for SPDX identifiers in {project_root}...")
    print(f"Excluding: {', '.join(EXCLUDES)}")

    failed_files = []
    checked_count = 0

    for root, dirs, files in os.walk(project_root):
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDES]

        for file in files:
            file_path = Path(root) / file
            if file_path.suffix in EXTENSIONS:
                checked_count += 1
                if not check_file(file_path):
                    failed_files.append(str(file_path.relative_to(project_root)))

    print(f"Checked {checked_count} files.")

    if failed_files:
        print("\nMissing SPDX-License-Identifier in:")
        for f in failed_files:
            print(f"  - {f}")
        return 1

    print("\nAll files have SPDX identifiers.")
    return 0

if __name__ == "__main__":
    sys.exit(main())

