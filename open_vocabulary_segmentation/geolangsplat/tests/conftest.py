# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shared test fixtures: CPU-only fakes so the suite needs no GPU/SAM3/fVDB."""
from __future__ import annotations

import types

import pytest
import torch

from geolangsplat.config import GeoLangSplatConfig


class FakeModel:
    """Stand-in for GaussianSplat3d holding only the tensors the writers touch."""

    def __init__(self, n: int = 12, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.means = torch.rand(n, 3, generator=g)
        self.quats = torch.rand(n, 4, generator=g)
        self.log_scales = torch.rand(n, 3, generator=g)
        self.logit_opacities = torch.rand(n, 1, generator=g)
        self.sh0 = torch.rand(n, 1, 3, generator=g)
        self.shN = torch.rand(n, 15, 3, generator=g)


class FakeEngine:
    """Minimal engine satisfying api.segment's contract, fully deterministic."""

    def __init__(self, cfg, model, n: int = 12, seed: int = 0):
        self.cfg = cfg
        self.model = model
        self.N = n
        g = torch.Generator().manual_seed(seed)
        self._scores = torch.rand(n, generator=g)
        self.denom = torch.ones(n)
        self.seen = torch.ones(n)

    def query(self, prompt, *, select=None, margin=None, compete=None):
        thr = self.cfg.select if select is None else select
        sel = self._scores >= thr
        return self._scores.clone(), sel, 0.001

    def bake_vocab(self, vocab):
        g = torch.Generator().manual_seed(123)
        return torch.rand(self.N, len(vocab), generator=g)


@pytest.fixture
def cfg():
    return GeoLangSplatConfig(device="cpu")


@pytest.fixture
def fake_model():
    return FakeModel(n=12)


@pytest.fixture
def fake_engine(cfg, fake_model):
    return FakeEngine(cfg, fake_model, n=12)
