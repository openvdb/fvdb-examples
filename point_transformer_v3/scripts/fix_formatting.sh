#!/bin/bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Exit on error
set -e

# Determine the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine the root of point_transformer_v3 (parent of scripts)
PTV3_ROOT="$(dirname "$DIR")"

# Change to the project root
cd "$PTV3_ROOT"

echo "Running black formatting on $(pwd)..."

# Run black, excluding the submodule
# The pattern "external/pointcept" will match the directory relative to the root
black --target-version=py311 --line-length=120 --extend-exclude "external/pointcept" .

echo "Formatting complete."
