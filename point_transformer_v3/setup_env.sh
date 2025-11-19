#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# Setup script for point_transformer_v3
# This sets up PYTHONPATH so imports work correctly

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/external/pointcept:${PYTHONPATH}"

echo "PYTHONPATH set to:"
echo "$PYTHONPATH"
echo ""
echo "You can now run scripts from this directory."
echo "Example: python minimal_inference.py --help"
