# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Enable `python -m geolangsplat ...` (used to spawn the background daemon)."""
from .cli import gls

if __name__ == "__main__":
    gls()
