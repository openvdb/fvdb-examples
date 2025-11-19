#!/usr/bin/env python3
"""
Setup script for point_transformer_v3 project.

This script sets up the Python path to allow imports from:
- fvdb_extensions (local extensions)
- external.pointcept.pointcept (pointcept submodule)

Usage:
    python setup_env.py
    # or source it:
    source setup_env.py  # This will export PYTHONPATH

Or import it in your scripts:
    import setup_env  # This will add paths to sys.path
"""

import os
import sys
from pathlib import Path

# Get the directory containing this script (point_transformer_v3)
PROJECT_ROOT = Path(__file__).parent.resolve()

def setup_paths():
    """Add necessary paths to sys.path for imports."""
    paths_to_add = [
        str(PROJECT_ROOT),  # For importing fvdb_extensions
        str(PROJECT_ROOT / "external" / "pointcept"),  # For importing pointcept
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return paths_to_add

def get_pythonpath():
    """Get PYTHONPATH string for shell export."""
    paths = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "external" / "pointcept"),
    ]
    return os.pathsep.join(paths)

if __name__ == "__main__":
    # When run as script, print export command
    pythonpath = get_pythonpath()
    print(f"export PYTHONPATH={pythonpath}:$PYTHONPATH")
    print("\n# Or run this script in Python to set up paths:")
    print("import setup_env")
else:
    # When imported, automatically set up paths
    setup_paths()

