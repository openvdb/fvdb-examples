# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Dual-head fusion math (instance + dense semantic) for the SAM3 scorer.

These exercise ``Sam3Scorer``'s head-fusion helpers on synthetic grounding outputs
so they run on CPU without loading SAM3 weights: we bypass ``__init__`` and set only
the attributes the fusion path reads.
"""
from types import SimpleNamespace

import torch

from geolangsplat.sam3 import Sam3Scorer


def _scorer(w=0.5, mode="mean", conf=0.2):
    s = Sam3Scorer.__new__(Sam3Scorer)
    s.proc = SimpleNamespace(confidence_threshold=conf)
    s.dual_head = True
    s.sem_weight = w
    s.sem_mode = mode
    return s


def _logits(*per_query):
    # SAM3 pred_logits is [B, Q, 1] (the trailing class dim is squeezed in decode).
    return torch.tensor([[[v] for v in per_query]])


def _outputs(pred_logits=None, presence=None, masks=None, sem=None):
    out = {}
    if pred_logits is not None:
        out["pred_logits"] = pred_logits
    if presence is not None:
        out["presence_logit_dec"] = presence
    if masks is not None:
        out["pred_masks"] = masks
    if sem is not None:
        out["semantic_seg"] = sem
    return out


def test_semantic_head_sigmoid_and_resize():
    s = _scorer()
    sem = torch.full((1, 1, 4, 4), 1.5)
    out = s._semantic_from_outputs(_outputs(sem=sem), 8, 8)
    assert out.shape == (8, 8)
    assert torch.allclose(out, torch.sigmoid(torch.tensor(1.5)).expand(8, 8), atol=1e-5)


def test_instance_head_presence_gate_and_keep():
    # presence ~sigmoid(2)=0.88; query0 survives sam_conf, query1 is dropped.
    s = _scorer(conf=0.5)
    presence = torch.tensor([2.0])
    pred_logits = _logits(2.0, -2.0)
    masks = torch.full((1, 2, 4, 4), 5.0)
    out = s._instance_from_outputs(_outputs(pred_logits, presence, masks), 4, 4)
    prob = torch.sigmoid(torch.tensor(2.0)) ** 2  # logit * presence
    expected = prob * torch.sigmoid(torch.tensor(5.0))  # mask prob
    assert out.shape == (4, 4)
    assert torch.allclose(out, expected.expand(4, 4), atol=1e-3)


def test_instance_head_empty_when_all_filtered():
    # presence ~0 -> every detection falls below sam_conf -> no instance.
    s = _scorer(conf=0.5)
    o = _outputs(_logits(5.0, 5.0), torch.tensor([-5.0]), torch.full((1, 2, 4, 4), 5.0))
    assert s._instance_from_outputs(o, 4, 4) is None


def test_dual_head_semantic_only_recall():
    # The recall case: instance head is filtered out, dense semantic head still fires.
    s = _scorer(w=0.5, mode="mean", conf=0.5)
    o = _outputs(
        _logits(5.0, 5.0),
        torch.tensor([-5.0]),
        torch.full((1, 2, 4, 4), 5.0),
        torch.full((1, 1, 4, 4), 2.0),
    )
    out = s._fuse_heads(o, 4, 4)
    expected = 0.5 * torch.sigmoid(torch.tensor(2.0))
    assert torch.allclose(out, expected.expand(4, 4), atol=1e-4)


def test_dual_head_mean_and_max_dispatch():
    s = _scorer(w=0.3, mode="mean", conf=0.2)
    o = _outputs(
        _logits(5.0),
        torch.tensor([5.0]),
        torch.full((1, 1, 4, 4), 3.0),
        torch.full((1, 1, 4, 4), -1.0),
    )
    inst = s._instance_from_outputs(o, 4, 4)
    sem = s._semantic_from_outputs(o, 4, 4)
    mean_out = s._fuse_heads(o, 4, 4)
    assert torch.allclose(mean_out, 0.7 * inst + 0.3 * sem, atol=1e-5)
    s.sem_mode = "max"
    max_out = s._fuse_heads(o, 4, 4)
    assert torch.allclose(max_out, torch.maximum(0.7 * inst, 0.3 * sem), atol=1e-5)


def test_dual_head_both_empty_returns_none():
    s = _scorer(conf=0.5)
    o = _outputs(_logits(-9.0), torch.tensor([-9.0]), torch.zeros(1, 1, 4, 4))
    assert s._fuse_heads(o, 4, 4) is None  # no semantic_seg key, instance filtered
