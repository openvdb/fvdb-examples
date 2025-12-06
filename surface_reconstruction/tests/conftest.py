# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration file.

Note: This project should be installed in editable mode with:
    pip install -e .

This makes all imports work normally without path manipulation.
"""

import sys
from unittest.mock import MagicMock


# Helper to mock a module and its submodules
def mock_module(module_name):
    if module_name not in sys.modules:
        m = MagicMock()
        m.__path__ = []  # Make it look like a package
        sys.modules[module_name] = m
    return sys.modules[module_name]


# Mock heavy frameworks that may not be installed in all test environments
mock_module("open3d")
