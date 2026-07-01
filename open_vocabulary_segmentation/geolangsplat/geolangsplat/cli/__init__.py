# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""GeoLangSplat command-line interface.

``gls`` dispatches subcommands, mirroring ``frgs``. The commands subclass the
shared ``BaseCommand`` so they can be registered directly inside ``frgs`` later
(e.g. as ``frgs segment "house"``).
"""
from __future__ import annotations

import sys
import time

import tyro

from .. import _PROCESS_T0
from ..errors import GeoLangSplatError
from ._bake import Bake
from ._catalog import Catalog
from ._check import Check
from ._common import BaseCommand
from ._doctor import Doctor
from ._explore import Explore
from ._render import Render
from ._segment import Segment
from ._serve import Serve
from ._show import Show
from ._status import Status
from ._stop import Stop

__all__ = [
    "gls",
    "Segment",
    "Bake",
    "Catalog",
    "Check",
    "Doctor",
    "Serve",
    "Show",
    "Explore",
    "Render",
    "Status",
    "Stop",
    "BaseCommand",
]


def gls() -> None:
    cmd: BaseCommand = tyro.cli(
        Segment | Bake | Catalog | Check | Doctor | Serve | Show | Explore | Render | Status | Stop
    )
    try:
        cmd.execute()
    except GeoLangSplatError as e:  # user-actionable: print the message, no traceback
        print(f"[gls] {e}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if getattr(cmd, "profile", False):
            print(
                f"[gls] total wall {time.perf_counter() - _PROCESS_T0:.1f}s "
                "(includes python/torch import + CUDA init)",
                flush=True,
            )
