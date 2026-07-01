# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls status` - show any warm engines running on this machine.

Like ``docker ps`` / ``ollama ps``: a quick answer to "is anything still loaded,
and how do I stop it?" so a background engine is never a mystery. One-shot
``gls segment`` keeps no state, so if nothing shows here, nothing is running.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ipc import cleanup_stale, list_sockets, request_status
from ._common import BaseCommand


def _fmt_secs(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m{s % 60:02d}s"


@dataclass
class Status(BaseCommand):
    """List running warm engines (and how to stop them)."""

    def execute(self) -> None:
        running = []
        for sock in list_sockets():
            st = request_status(sock)
            if st:
                running.append((sock, st))
            else:
                cleanup_stale(sock)  # drop the socket file of a dead daemon

        if not running:
            print("[status] no warm engines running.")
            print("[status] start one:  gls serve <model.ply> -b")
            return

        print(f"[status] {len(running)} warm engine(s) running:\n")
        for _sock, st in running:
            model = st.get("model", "?")
            print(f"  {model}")
            print(
                f"      pid {st.get('pid')}  |  {st.get('N', 0):,} gaussians" f"  |  recipe {st.get('recipe') or '-'}"
            )
            print(
                f"      up {_fmt_secs(st.get('uptime_s'))}"
                f"  |  idle {_fmt_secs(st.get('idle_s'))}"
                f"  |  auto-stop in {_fmt_secs(st.get('idle_remaining_s'))}"
            )
            if st.get("vram"):
                print(f"      {st['vram']}")
            print(f"      stop:  gls stop {model}")
        print("\n[status] stop everything:  gls stop")
