# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Checks against the installed fVDB packages, with no fakes.

The rest of the suite mocks fVDB, so it cannot catch API moves. These tests skip
when fvdb-reality-capture is not installed.
"""
import importlib
import pathlib
import pkgutil

import pytest
import torch

frc = pytest.importorskip("fvdb_reality_capture")

import geolangsplat
import geolangsplat.outputs as out

_PKG_DIR = pathlib.Path(geolangsplat.__file__).parent
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_all_modules_import():
    for mod in pkgutil.walk_packages([str(_PKG_DIR)], prefix="geolangsplat."):
        importlib.import_module(mod.name)


def test_no_removed_fvdb_imports():
    offenders = [
        str(p.relative_to(_PKG_DIR))
        for p in _PKG_DIR.rglob("*.py")
        if "from fvdb import GaussianSplat3d" in p.read_text()
    ]
    assert offenders == []


def test_writer_uses_reality_capture_splat():
    assert out._gs() is frc.GaussianSplat3d


def test_reality_capture_entry_points_exist():
    from fvdb_reality_capture.radiance_fields import load_splats_from_file  # noqa: F401

    assert callable(frc.gaussian_splat_to_view_data)


def _random_splat(n: int = 64):
    g = torch.Generator().manual_seed(0)
    q = torch.rand(n, 4, generator=g)
    return frc.GaussianSplat3d.from_tensors(
        means=torch.rand(n, 3, generator=g).cuda(),
        quats=(q / q.norm(dim=1, keepdim=True)).cuda(),
        log_scales=torch.full((n, 3), -4.0).cuda(),
        logit_opacities=torch.zeros(n).cuda(),
        sh0=torch.rand(n, 1, 3, generator=g).cuda(),
        shN=torch.zeros(n, 15, 3).cuda(),
    )


@_CUDA
def test_ply_writers_round_trip(tmp_path):
    model = _random_splat()
    sel = torch.zeros(model.num_gaussians, dtype=torch.bool, device="cuda")
    sel[:10] = True

    out.write_ply_segmented(model, sel, tmp_path / "seg.ply")
    out.write_ply_overlay(model, sel, tmp_path / "overlay.ply")

    seg, _ = frc.GaussianSplat3d.from_ply(tmp_path / "seg.ply")
    overlay, _ = frc.GaussianSplat3d.from_ply(tmp_path / "overlay.ply")
    assert seg.num_gaussians == 10
    assert overlay.num_gaussians == model.num_gaussians
