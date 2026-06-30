# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Typed errors so the CLI can print a clean, actionable message (no traceback)
for the failures users actually hit: no GPU, missing SAM3 weights, bad inputs."""
from __future__ import annotations


class GeoLangSplatError(RuntimeError):
    """A user-actionable error (bad device, missing weights, bad input).

    The ``gls`` CLI catches this and prints only the message; everything else
    propagates as a normal traceback for debugging.
    """
