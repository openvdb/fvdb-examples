# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for the step profiler (CPU-only)."""
from __future__ import annotations

import time

from geolangsplat.profile import Stopwatch


def test_stopwatch_records_and_reports():
    sw = Stopwatch(device=None)  # cpu: no cuda sync
    with sw.span("a"):
        time.sleep(0.01)
    sw.add("b", 0.5)
    names = [n for n, _ in sw.spans]
    assert names == ["a", "b"]
    assert sw.total() >= 0.5
    rep = sw.report("build")
    assert "build" in rep
    assert "a" in rep and "b" in rep
    assert "TOTAL" in rep


def test_stopwatch_empty_report_is_blank():
    assert Stopwatch(device=None).report() == ""
