# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls stop` - shut down a warm engine started by `gls serve`.

With a model path, stops that engine. With no arguments, stops every warm engine
on this machine (handy when you forgot what's running -- see `gls status`).
Shutdown is guaranteed: polite IPC first, then SIGTERM/SIGKILL, then the socket
and pidfile are cleaned up either way.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional

import tyro

from ..ipc import default_socket, kill_daemon, list_sockets
from ._common import BaseCommand


@dataclass
class Stop(BaseCommand):
    """Stop a warm background engine, or all of them if no model is given."""

    # Splat the daemon was started with (omit to stop ALL running engines).
    model_path: tyro.conf.Positional[Optional[pathlib.Path]] = None

    # Explicit Unix socket path (overrides the per-model default).
    socket_path: Optional[str] = None

    def execute(self) -> None:
        if self.model_path is None and not self.socket_path:
            self._stop_all()
            return
        sock = self.socket_path or default_socket(self.model_path)
        # kill_daemon guarantees the process is gone (IPC -> SIGTERM -> SIGKILL)
        # and removes the socket + pidfile either way.
        status = kill_daemon(sock)
        print(f"[stop] {status} ({self.model_path or sock})", flush=True)

    def _stop_all(self) -> None:
        stopped = 0
        for sock in list_sockets():
            status = kill_daemon(sock)
            if status in ("stopped", "killed"):
                stopped += 1
        print(f"[stop] stopped {stopped} engine(s)" if stopped else "[stop] no engines were running", flush=True)
