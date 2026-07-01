# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Robustness/edge-case tests: device resolution, missing weights, metadata
round-trip, amp fallback. All CPU-only (no GPU/SAM3/fvdb needed)."""
from __future__ import annotations

import pytest
import torch

import geolangsplat.outputs as out
from geolangsplat.engine import _load_model, resolve_device
from geolangsplat.errors import GeoLangSplatError
from geolangsplat.sam3 import Sam3Scorer, _resolve_amp_dtype


# -- device resolution ------------------------------------------------------


def test_resolve_device_cpu_ok():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_without_gpu_raises():
    if torch.cuda.is_available():
        pytest.skip("CUDA present; cannot test the no-GPU path here")
    with pytest.raises(GeoLangSplatError):
        resolve_device("cuda")


# -- model loading ----------------------------------------------------------


def test_load_model_missing_file_raises():
    with pytest.raises(GeoLangSplatError):
        _load_model("/no/such/model.ply", torch.device("cpu"))


def test_load_model_passthrough_for_object():
    sentinel = object()
    model, meta = _load_model(sentinel, torch.device("cpu"))
    assert model is sentinel and meta == {}


# -- SAM3 guards ------------------------------------------------------------


def test_sam3_missing_checkpoint_raises():
    with pytest.raises(GeoLangSplatError):
        Sam3Scorer("/no/such/sam3.pt")


def test_amp_dtype_fp16_and_default():
    assert _resolve_amp_dtype("fp16", "cuda") == torch.float16
    # on cpu (no cuda bf16 query) the default stays bf16
    assert _resolve_amp_dtype("bf16", "cpu") == torch.bfloat16


# -- metadata round-trip ----------------------------------------------------


class _FakeGS:
    last_metadata = "UNSET"

    @classmethod
    def from_tensors(cls, **kw):
        inst = cls()
        inst.kw = kw
        return inst

    def save_ply(self, path, metadata=None):
        _FakeGS.last_metadata = metadata
        with open(path, "w") as f:
            f.write("ok\n")


def test_writer_passes_metadata(monkeypatch, fake_model, tmp_path):
    monkeypatch.setattr(out, "_gs", lambda: _FakeGS)
    sel = torch.zeros(12, dtype=torch.bool)
    sel[:3] = True
    out.write_ply_overlay(fake_model, sel, tmp_path / "o.ply", metadata={"foo": "bar"})
    assert _FakeGS.last_metadata == {"foo": "bar"}


class _PickyGS(_FakeGS):
    def save_ply(self, path, metadata=None):
        if metadata:  # backend rejects rich metadata -> writer must retry without it
            raise ValueError("unsupported metadata")
        with open(path, "w") as f:
            f.write("ok\n")


def test_writer_falls_back_when_metadata_rejected(monkeypatch, fake_model, tmp_path):
    monkeypatch.setattr(out, "_gs", lambda: _PickyGS)
    sel = torch.zeros(12, dtype=torch.bool)
    sel[:2] = True
    p = tmp_path / "o.ply"
    out.write_ply_segmented(fake_model, sel, p, metadata={"nested": {"x": 1}})
    assert p.exists()  # written despite metadata rejection
