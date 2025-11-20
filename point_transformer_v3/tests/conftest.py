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
        m.__path__ = [] # Make it look like a package
        sys.modules[module_name] = m
    return sys.modules[module_name]

# Mock pointops to avoid installing custom CUDA extensions for unit tests
mock_module("pointops")
mock_module("pointgroup_ops")

# Mock heavy frameworks
# Note: We must mock parent packages before children for some import mechanisms to be happy
tg = mock_module("torch_geometric")
mock_module("torch_geometric.utils")
mock_module("torch_geometric.nn")
mock_module("torch_geometric.nn.pool")

mock_module("torch_scatter")
mock_module("torch_sparse")
mock_module("open3d")
