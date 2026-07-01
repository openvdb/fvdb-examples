# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Thin SAM3 wrapper: build the image model, cache per-view image embeddings,
and turn a text prompt into a per-pixel score map.

SAM3 is mixed precision and must run under autocast. The class is duck-typed so
tests can substitute a lightweight stand-in (anything exposing ``encode`` and
``scoremap``); see ``tests/conftest.py``.
"""
from __future__ import annotations

import contextlib
import os
import pathlib
import warnings
from typing import Any

import torch
import torch.nn.functional as F

from .config import DEFAULT_DISTRACTORS
from .errors import GeoLangSplatError


@contextlib.contextmanager
def _quiet_load():
    """Silence SAM3's import/load chatter (timm/pkg_resources FutureWarnings and the
    model loader's ``missing_keys`` dump) while still letting real exceptions raise."""
    with (
        warnings.catch_warnings(),
        open(os.devnull, "w") as _dn,
        contextlib.redirect_stdout(_dn),
        contextlib.redirect_stderr(_dn),
    ):
        warnings.simplefilter("ignore")
        yield


def encode_progress(done: int, total: int, label: str = "encode") -> None:
    """Draw a single in-place progress line (carriage return, no newline)."""
    width = 24
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{label}] [{bar}] {done}/{total} views", end="", flush=True)


def encode_progress_done() -> None:
    """Terminate a progress line started by :func:`encode_progress`."""
    print(flush=True)


def _resolve_amp_dtype(amp: str, device: str) -> torch.dtype:
    """bf16 by default, but fall back to fp16 on GPUs that lack bf16 support."""
    want = amp.lower()
    if want == "fp16":
        return torch.float16
    if str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            if not torch.cuda.is_bf16_supported():
                print("[sam3] bf16 unsupported on this GPU -> using fp16", flush=True)
                return torch.float16
        except Exception:
            pass
    return torch.bfloat16


class Sam3Scorer:
    """Wraps a SAM3 image model + processor for promptable per-pixel scoring."""

    def __init__(
        self,
        sam_ckpt: str,
        sam_res: int = 1008,
        sam_conf: float = 0.20,
        amp: str = "bf16",
        device: str = "cuda",
        dual_head: bool = False,
        sem_weight: float = 0.5,
        sem_mode: str = "mean",
    ):
        if not str(sam_ckpt).strip():
            raise GeoLangSplatError(
                "SAM3 checkpoint is not set. Point GeoLangSplat at the weights via the "
                "GEOLANGSPLAT_SAM_CKPT environment variable, e.g.\n"
                "  export GEOLANGSPLAT_SAM_CKPT=/path/to/sam3.1_multiplex.pt\n"
                "(or pass GeoLangSplatConfig(sam_ckpt=...)). Geometry-only `gls check` needs no weights."
            )
        if not pathlib.Path(sam_ckpt).exists():
            raise GeoLangSplatError(
                f"SAM3 checkpoint not found: {sam_ckpt}\n"
                "Set it via GEOLANGSPLAT_SAM_CKPT or GeoLangSplatConfig(sam_ckpt=...). SAM3 weights "
                "are required for segmentation; geometry-only `gls check` does not need them."
            )
        try:
            with _quiet_load():
                from sam3.model.sam3_image_processor import Sam3Processor
                from sam3.model_builder import build_sam3_image_model
        except Exception as ex:  # SAM3 package not installed in this env
            raise GeoLangSplatError(
                f"the `sam3` package is required for segmentation but could not be imported ({ex}).\n"
                "Install SAM3 in this environment, or use geometry-only `gls check`."
            ) from ex

        with _quiet_load():
            model = build_sam3_image_model(device=device, checkpoint_path=sam_ckpt, load_from_HF=False)
            self.proc = Sam3Processor(model, resolution=sam_res, device=device, confidence_threshold=sam_conf)
        self.model = model
        self.amp_dtype = _resolve_amp_dtype(amp, device)
        self.device = device
        self.dual_head = bool(dual_head)  # fuse the dense semantic head into the score map
        self.sem_weight = float(sem_weight)
        self.sem_mode = str(sem_mode)
        self._dummy_prompt = None  # cached image-independent dummy geometric prompt
        self._batch_ok = None  # tri-state: None=unverified, True/False=cached split-vs-ref check

    # -- embedding cache ----------------------------------------------------

    @staticmethod
    def _split_backbone(bb: Any, i: int) -> Any:
        """Index batch element ``i`` out of a (possibly nested) backbone_out."""
        if isinstance(bb, dict):
            return {k: Sam3Scorer._split_backbone(v, i) for k, v in bb.items()}
        if isinstance(bb, (list, tuple)):
            return type(bb)(Sam3Scorer._split_backbone(v, i) for v in bb)
        if torch.is_tensor(bb):
            if bb.dim() >= 1 and bb.shape[0] > i:
                return bb[i : i + 1].contiguous()
            return bb
        return bb

    def _verify_batch_split(self, pils, height: int, width: int) -> bool:
        """One-time check that indexing a batched encode matches the per-view encode.

        Cached on ``self._batch_ok`` so it runs ONCE per process, not per encode call.
        The streaming lift encodes in many small chunks, and re-validating every chunk
        was ~3 wasted encodes each -- the dominant overhead vs. the warm build (which
        validates once). Correctness still never depends on the batched split: if the
        check fails we fall back to per-view encoding everywhere.
        """
        if self._batch_ok is not None:
            return self._batch_ok
        try:
            sb = self.proc.set_image_batch(pils[:2])
            v1 = {
                "backbone_out": self._split_backbone(sb["backbone_out"], 1),
                "original_height": height,
                "original_width": width,
            }
            ref1 = self.proc.set_image(pils[1])
            a = self.scoremap("building", v1)
            b = self.scoremap("building", ref1)
            self._batch_ok = (a is None and b is None) or (
                a is not None
                and b is not None
                and a.shape == b.shape
                and torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)
            )
        except Exception as ex:
            print(f"[encode] batched encode unavailable ({ex}) -> per-view", flush=True)
            self._batch_ok = False
        return self._batch_ok

    @torch.inference_mode()
    def encode(
        self, pils, height: int, width: int, batch_encode: bool = True, batch_size: int = 8, progress: bool = False
    ):
        """Run the SAM3 image encoder once per view and cache the result.

        Tries a batched encode for a faster build; the batched-split-vs-per-view check
        is validated once per process (:meth:`_verify_batch_split`) and on any mismatch
        or error it falls back to per-view, so correctness never depends on batching.
        ``progress`` draws a single in-place progress line over the encode (used by the
        warm build; the streaming lift draws its own across chunks).
        """
        n = len(pils)
        if n == 0:
            return []
        with torch.autocast("cuda", dtype=self.amp_dtype):
            if batch_encode and n > 1 and self._verify_batch_split(pils, height, width):
                states: list = [None] * n
                for c0 in range(0, n, batch_size):
                    chunk = pils[c0 : c0 + batch_size]
                    sb = self.proc.set_image_batch(chunk)
                    for j in range(len(chunk)):
                        states[c0 + j] = {
                            "backbone_out": self._split_backbone(sb["backbone_out"], j),
                            "original_height": height,
                            "original_width": width,
                        }
                    if progress:
                        encode_progress(min(c0 + batch_size, n), n)
                if progress:
                    encode_progress_done()
                return states
            states = []
            for k, p in enumerate(pils):
                states.append(self.proc.set_image(p))
                if progress:
                    encode_progress(k + 1, n)
            if progress:
                encode_progress_done()
        return states

    # -- scoring ------------------------------------------------------------

    def forward_text(self, prompt: str):
        """Encode the text prompt once (image-independent); reuse across views."""
        return self.proc.model.backbone.forward_text([prompt], device=self.device)

    def _decode(self, prompt: str, state: dict, text_outputs=None, out_hw=None):
        """Run SAM3's grounding decode for ``prompt`` on a cached embedding and return
        the raw output dict (``masks_logits`` [K,1,h,w] probs, ``scores`` [K], ...).

        Critically we decode on a *throwaway* working dict, not on ``state`` itself:
        ``_forward_grounding`` writes the (full-stack) mask tensors back into the dict
        it is given, and ``state`` lives in the resident per-view embedding cache.
        Mutating it would keep every view's masks alive for the whole query (they would
        accumulate across all views instead of being freed per view) -- the original
        cause of the runaway query VRAM.
        """
        if text_outputs is None:
            text_outputs = self.forward_text(prompt)
        if self._dummy_prompt is None:
            self._dummy_prompt = self.proc.model._get_dummy_prompt()
        backbone_out = dict(state["backbone_out"])  # shallow copy: shares tensors, new keys are local
        backbone_out.update(text_outputs)
        work = {
            "backbone_out": backbone_out,
            "original_height": out_hw[0] if out_hw else state["original_height"],
            "original_width": out_hw[1] if out_hw else state["original_width"],
            "geometric_prompt": self._dummy_prompt,
        }
        return self.proc._forward_grounding(work)

    def scoremap(self, prompt: str, state: dict, text_outputs=None, out_hw=None):
        """Text + grounding decode on a cached image embedding -> ``[H, W]`` scores
        (or ``None`` if SAM3 returns no masks).

        ``out_hw=(h, w)`` decodes the masks straight to that size instead of the
        original photo resolution. The alpha lift downsamples to its lift
        resolution anyway, so passing the lift size here avoids upsampling every
        instance mask to full photo res and then throwing it away -- a large
        memory + time saving on high-resolution aerial captures.

        With ``dual_head`` the dense semantic head is fused in (see
        :meth:`_fuse_heads`); otherwise this is the presence-gated instance head only.
        """
        if self.dual_head:
            h = out_hw[0] if out_hw else state["original_height"]
            w = out_hw[1] if out_hw else state["original_width"]
            outputs = self._grounding_outputs(prompt, state, text_outputs)
            return self._fuse_heads(outputs, h, w)
        st = self._decode(prompt, state, text_outputs, out_hw)
        ml = st.get("masks_logits")
        if ml is None or ml.shape[0] == 0:
            return None
        sc = st["scores"]
        return (ml[:, 0] * sc.view(-1, 1, 1)).amax(0).float()  # [H, W]

    # -- dual-head (instance + semantic) fusion -----------------------------

    def _grounding_outputs(self, prompt: str, state: dict, text_outputs=None) -> dict:
        """Run SAM3's grounding model once and return its *raw* output dict.

        Unlike :meth:`_decode` (which routes through the processor's
        ``_forward_grounding`` and keeps only instance masks/scores), this keeps the
        full output -- including ``semantic_seg`` -- so both heads come from a single
        forward. As in :meth:`_decode`, we build a throwaway working ``backbone_out``
        so we never mutate the resident per-view embedding cache.
        """
        if text_outputs is None:
            text_outputs = self.forward_text(prompt)
        if self._dummy_prompt is None:
            self._dummy_prompt = self.proc.model._get_dummy_prompt()
        backbone_out = dict(state["backbone_out"])  # shallow copy: shares tensors
        backbone_out.update(text_outputs)
        return self.proc.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.proc.find_stage,
            geometric_prompt=self._dummy_prompt,
            find_target=None,
        )

    def _instance_from_outputs(self, outputs: dict, img_h: int, img_w: int):
        """Presence-gated instance head -> ``[H, W]`` in [0, 1] (or ``None``).

        Mirrors the processor's ``_forward_grounding`` exactly (presence multiply +
        ``sam_conf`` keep), minus the box bookkeeping we don't need.
        """
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence).squeeze(-1)
        keep = out_probs > self.proc.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        if out_masks.shape[0] == 0:
            return None
        out_masks = F.interpolate(
            out_masks.unsqueeze(1).float(), (img_h, img_w), mode="bilinear", align_corners=False
        ).sigmoid()
        return (out_masks[:, 0] * out_probs.view(-1, 1, 1).float()).amax(0).float()

    def _semantic_from_outputs(self, outputs: dict, img_h: int, img_w: int):
        """Dense prompt-conditioned semantic head -> ``[H, W]`` in [0, 1] (or ``None``).

        ``semantic_seg`` is a single-channel logit map (``Conv2d(.., 1, 1)``); we take
        the first image, sigmoid, and resize to the requested resolution.
        """
        sem = outputs.get("semantic_seg")
        if sem is None:
            return None
        sem = sem.float()
        while sem.dim() < 4:  # [h,w]/[1,h,w] -> [1,1,h,w]
            sem = sem.unsqueeze(0)
        sem = F.interpolate(sem[:1], (img_h, img_w), mode="bilinear", align_corners=False).sigmoid()
        return sem[0, 0]

    def _fuse_heads(self, outputs: dict, img_h: int, img_w: int):
        """Blend the instance and semantic heads into one ``[H, W]`` score (or ``None``).

        Missing heads count as 0, so the blend is continuous when either head fires
        alone -- the semantic-only case (instance head filtered out by ``sam_conf``) is
        exactly the recall the dense head is meant to recover. ``sem_mode='mean'`` is a
        convex blend; ``'max'`` takes the stronger weighted head per pixel.
        """
        inst = self._instance_from_outputs(outputs, img_h, img_w)
        sem = self._semantic_from_outputs(outputs, img_h, img_w)
        if inst is None and sem is None:
            return None
        w = self.sem_weight
        if inst is None:
            inst = torch.zeros_like(sem)
        if sem is None:
            sem = torch.zeros_like(inst)
        if self.sem_mode == "max":
            return torch.maximum((1.0 - w) * inst, w * sem)
        return (1.0 - w) * inst + w * sem
