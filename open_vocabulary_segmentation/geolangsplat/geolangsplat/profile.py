# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""A tiny, always-on step profiler for the engine build / cache-load paths.

GPU work is asynchronous, so wall-clock around a CUDA call under-counts it. Each
span therefore ``torch.cuda.synchronize()``-es on enter and exit, giving an
honest per-step breakdown (load splat, render views, alpha lift, SAM3 load, SAM3
encode, cache read/restore). The overhead (a couple of syncs per major step) is
negligible next to the steps themselves, so it stays on by default -- the whole
point of this exercise is to know where the seconds go.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class Stopwatch:
    """Accumulates named, CUDA-synchronized spans and prints a breakdown."""

    def __init__(self, device=None, enabled: bool = True):
        self.device = device
        self.enabled = enabled
        self.spans: list[tuple[str, float]] = []
        self._cuda = bool(device is not None and getattr(device, "type", None) == "cuda")

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        if self._cuda:
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        try:
            yield
        finally:
            if self._cuda:
                torch.cuda.synchronize(self.device)
            self.spans.append((name, time.perf_counter() - t))

    def add(self, name: str, seconds: float) -> None:
        """Record a pre-measured span (for time spent outside a ``span`` block)."""
        self.spans.append((name, float(seconds)))

    def total(self) -> float:
        return sum(s for _, s in self.spans)

    def report(self, title: str = "profile") -> str:
        if not self.spans:
            return ""
        tot = self.total() or 1e-9
        w = max(len(n) for n, _ in self.spans)
        lines = [f"[{title}] step breakdown (cuda-synced):"]
        for n, s in self.spans:
            bar = "#" * int(round(28 * s / tot))
            lines.append(f"    {n.ljust(w)}  {s:8.2f}s  {100 * s / tot:5.1f}%  {bar}")
        lines.append(f"    {'TOTAL'.ljust(w)}  {tot:8.2f}s")
        return "\n".join(lines)
