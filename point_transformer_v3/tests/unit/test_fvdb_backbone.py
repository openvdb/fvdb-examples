# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the PT-v3fvdb Pointcept backbone, which wraps fvdb.nn.PointTransformerV3.
"""

import importlib.util
import unittest

import torch

import fvdb.nn as fvnn

HAS_CORE_MODEL = hasattr(fvnn, "PointTransformerV3")
HAS_VARLEN = importlib.util.find_spec("torch.nn.attention.varlen") is not None
CUDA = torch.cuda.is_available()

SMALL_CONFIG = dict(
    in_channels=6,
    order=("z", "z-trans"),
    stride=(2, 2),
    enc_depths=(1, 1, 1),
    enc_channels=(16, 32, 64),
    enc_num_head=(1, 2, 4),
    dec_depths=(1, 1),
    dec_channels=(16, 32),
    dec_num_head=(1, 2),
    embedding_mode="linear",
    norm="batch",
)


def _point_batch(sizes=(3000, 2000), extent=40, seed=0):
    """Pointcept-style batch with duplicate coordinates, as grid sampling with test mode can produce."""
    gen = torch.Generator().manual_seed(seed)
    coords = [torch.randint(0, extent, (n, 3), generator=gen) for n in sizes]
    grid_coord = torch.cat(coords).cuda()
    feat = torch.randn(len(grid_coord), 6, generator=gen).cuda()
    offset = torch.cumsum(torch.tensor(sizes), 0).cuda()
    return dict(grid_coord=grid_coord, feat=feat, offset=offset)


@unittest.skipUnless(CUDA and HAS_CORE_MODEL, "requires CUDA and fvdb with fvdb.nn.PointTransformerV3")
class TestFvdbBackbone(unittest.TestCase):
    def setUp(self):
        from fvdb_extensions.models.point_transformer_v3m1_fvdb import PointTransformerV3, create_grid_from_points

        self.backbone_cls = PointTransformerV3
        self.create_grid_from_points = create_grid_from_points
        torch.manual_seed(0)

    def test_points_map_to_their_voxels(self):
        batch = _point_batch()
        grid, voxel_feat, point_to_voxel = self.create_grid_from_points(**batch)
        self.assertEqual(grid.grid_count, 2)
        self.assertEqual(point_to_voxel.shape, (len(batch["grid_coord"]),))
        self.assertTrue(torch.equal(grid.ijk.jdata[point_to_voxel].long(), batch["grid_coord"]))
        self.assertEqual(voxel_feat.jdata.shape, (grid.total_voxels, 6))

    def test_forward_backward(self):
        patch = 64 if HAS_VARLEN else 0
        backbone = self.backbone_cls(**SMALL_CONFIG, enc_patch_size=patch, dec_patch_size=patch).cuda()
        batch = _point_batch()
        out = backbone(batch)
        self.assertEqual(out.shape, (len(batch["grid_coord"]), 16))
        self.assertTrue(bool(torch.isfinite(out).all()))
        out.square().mean().backward()
        for name, param in backbone.named_parameters():
            self.assertIsNotNone(param.grad, name)

    def test_pointcept_options_forward(self):
        patch = 64 if HAS_VARLEN else 0
        backbone = self.backbone_cls(
            **SMALL_CONFIG,
            enc_patch_size=patch,
            dec_patch_size=patch,
            serialization_offset="none",
            pointcept_unpool_cpe=True,
        ).cuda()
        batch = _point_batch()
        out = backbone(batch)
        self.assertEqual(out.shape, (len(batch["grid_coord"]), 16))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_cls_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.backbone_cls(**SMALL_CONFIG, cls_mode=True)


if __name__ == "__main__":
    unittest.main()
