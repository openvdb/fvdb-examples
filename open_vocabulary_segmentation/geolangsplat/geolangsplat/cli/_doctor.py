# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls doctor` - one-command environment preflight.

Validates everything `gls segment`/`bake` needs on a fresh machine -- GPU, fvdb,
SAM3, the SAM3 checkpoint, bf16 support -- and prints actionable hints for whatever
is missing, instead of failing deep inside a build. (SAM3 weights + a working CUDA
stack are the usual pain when moving to a new box.)
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
from dataclasses import dataclass

from ._common import BaseCommand

_OK = "[ ok ]"
_FAIL = "[FAIL]"
_WARN = "[warn]"


@dataclass
class Doctor(BaseCommand):
    """Check that this machine can actually run GeoLangSplat segmentation."""

    def execute(self) -> None:
        from .. import __version__

        hard_ok = True
        print(f"[doctor] GeoLangSplat v{__version__} environment check\n")

        # --- torch + CUDA ---------------------------------------------------
        try:
            import torch

            print(f"{_OK} torch {torch.__version__}")
            if torch.cuda.is_available():
                n = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
                print(f"{_OK} CUDA available: {n} device(s); cuda:0 = {name} (sm_{cap})")
                bf16 = torch.cuda.is_bf16_supported()
                mark = _OK if bf16 else _WARN
                print(
                    f"{mark} bf16 {'supported' if bf16 else 'unsupported -> SAM3 falls back to fp16 (slower/less stable)'}"
                )
            else:
                hard_ok = False
                print(f"{_FAIL} CUDA not available -- segmentation needs a GPU")
                print(
                    "       check drivers / `nvidia-smi` / CUDA_VISIBLE_DEVICES "
                    "(geometry-only `gls check` still works on CPU)"
                )
        except Exception as e:
            hard_ok = False
            print(f"{_FAIL} torch import failed: {e}")

        # --- fvdb -----------------------------------------------------------
        if importlib.util.find_spec("fvdb") is not None:
            print(f"{_OK} fvdb importable")
        else:
            hard_ok = False
            print(f"{_FAIL} fvdb not importable -- activate the env that has the fvdb build")

        # --- fvdb-reality-capture (frgs integration / checkpoint IO) --------
        if importlib.util.find_spec("fvdb_reality_capture") is not None:
            print(f"{_OK} fvdb_reality_capture importable (frgs IO + .pt/.pth checkpoints)")
        else:
            print(
                f"{_WARN} fvdb_reality_capture not importable -- .ply still works; "
                ".pt/.pth checkpoint loading + frgs drop-in unavailable"
            )

        # --- SAM3 -----------------------------------------------------------
        if importlib.util.find_spec("sam3") is not None:
            print(f"{_OK} sam3 importable")
        else:
            hard_ok = False
            print(
                f"{_FAIL} sam3 not importable -- install SAM3 from source and put it on PYTHONPATH "
                "(needed for segment/bake; not for check)"
            )

        # --- SAM3 checkpoint ------------------------------------------------
        ckpt = os.environ.get("GEOLANGSPLAT_SAM_CKPT", "").strip()
        if not ckpt:
            hard_ok = False
            print(f"{_FAIL} GEOLANGSPLAT_SAM_CKPT not set")
            print("       export GEOLANGSPLAT_SAM_CKPT=/path/to/sam3.1_multiplex.pt " "(or pass --sam-ckpt)")
        elif not pathlib.Path(ckpt).exists():
            hard_ok = False
            print(f"{_FAIL} GEOLANGSPLAT_SAM_CKPT points to a missing file: {ckpt}")
        else:
            size_gb = pathlib.Path(ckpt).stat().st_size / 1e9
            print(f"{_OK} SAM3 checkpoint: {ckpt} ({size_gb:.1f} GB)")

        print()
        if hard_ok:
            print("[doctor] all required checks passed -- `gls segment` should run here.")
        else:
            print("[doctor] one or more required checks FAILED -- fix the [FAIL] lines above.")
            raise SystemExit(1)
