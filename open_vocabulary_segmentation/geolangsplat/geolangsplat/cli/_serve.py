# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""`gls serve` - build a warm engine once, then query it with `gls segment`.

Builds the scene cache (load + views + SAM3 embeddings) ONCE and listens on a
Unix socket; query it afterwards with plain ``gls segment`` (which auto-detects
the warm engine and answers instantly). This is the "engine" way to use
GeoLangSplat: ``gls serve`` -> ``gls segment`` (many times) -> ``gls stop``. The
engine auto-stops after ``--keep-alive`` idle minutes so it never lingers.
"""
from __future__ import annotations

import atexit
import os
import pathlib
import signal
import socket
import time
from dataclasses import dataclass
from typing import Annotated, Optional

import tyro
from tyro.conf import arg

from ..config import RecipeName, apply_recipe
from ._common import resolve_recipe
from ..ipc import (
    DEFAULT_KEEP_ALIVE_MIN,
    default_socket,
    handle_request,
    recv_obj,
    remove_runtime_files,
    send_obj,
    write_pidfile,
)
from ._common import BaseCommand, build_config


@dataclass
class Serve(BaseCommand):
    """
    Start a warm engine daemon, then query it with `gls segment`.

    Example:

        gls serve   model.ply --recipe satellite -b      # build once, detach
        gls segment model.ply "building" -O ply_overlay -o b.ply   # auto-uses it
        gls stop    model.ply
    """

    # Path to the input Gaussian splat .ply.
    model_path: tyro.conf.Positional[pathlib.Path]

    # Capture-type preset: satellite | satellite_dense | aerial.
    recipe: Annotated[Optional[RecipeName], arg(aliases=["-r"])] = None

    # View source: render (synthetic orbit/ladder) | globe (dome) | images (real photos).
    view_source: Optional[str] = None

    # Cap the synthesized/subsampled view count (build VRAM & query latency ~ views).
    max_views: Optional[int] = None

    # If >0, trim the view plan to fit this VRAM budget (GB).
    vram_budget_gb: Optional[float] = None

    # Scene up axis: auto (estimate from ground plane) or +z,-z,+y,-y,+x,-x.
    up: Annotated[Optional[str], arg(aliases=["-u"])] = None

    # Path to the SfM/COLMAP scene (required for --view-source images).
    sfm: Annotated[str, arg(aliases=["-s"])] = ""

    # Default score threshold for this session (per-query --select still wins).
    select: Annotated[Optional[float], arg(aliases=["-t"])] = None

    # Default concept competition for this session.
    compete: bool = False

    # Fast queries: aggregate only this many angularly-spread views per query (0 = all
    # views, full quality). The grounding decode is the per-query cost, so k of N views
    # is ~N/k faster -- a session-wide interactive speed lever (e.g. for DGX Spark).
    fast_views: Optional[int] = None

    # Launch detached in the background and return the terminal (instead of blocking).
    background: Annotated[bool, arg(aliases=["-b"])] = False

    # Auto-stop after this many idle minutes (sliding; reset only by real queries,
    # not status checks). 0 or negative = stay up until `gls stop`. The one knob
    # for the engine's lifetime -- a forgotten daemon frees the GPU on its own.
    keep_alive: float = DEFAULT_KEEP_ALIVE_MIN

    # Device.
    device: Annotated[str, arg(aliases=["-d"])] = "cuda"

    # Advanced: socket the daemon binds (auto-derived from the model path; the
    # warm-spawn path passes this through, so it must remain a real CLI option).
    socket_path: Optional[str] = None

    def execute(self) -> None:
        from ..engine import load_or_build_engine
        from ..ipc import daemon_alive, spawn_daemon, wait_ready

        if not self.model_path.exists():
            print(f"[serve] no such file: {self.model_path}", flush=True)
            return

        sock_path = self.socket_path or default_socket(self.model_path)
        recipe = resolve_recipe(self.recipe, self.view_source)

        if self.background:
            if daemon_alive(sock_path):
                print(f"[serve] already running for {self.model_path}", flush=True)
                return
            print("[serve] launching in background (streaming build below)...", flush=True)
            proc, log = spawn_daemon(
                self.model_path,
                recipe=recipe,
                sfm=self.sfm or None,
                view_source=self.view_source,
                max_views=self.max_views,
                vram_budget_gb=self.vram_budget_gb,
                up=self.up,
                select=self.select,
                compete=self.compete,
                fast_views=self.fast_views,
                device=self.device,
                socket_path=sock_path,
                keep_alive=self.keep_alive,
            )
            if wait_ready(sock_path, proc=proc, log_path=log):
                print(
                    f"[serve] ready.\n"
                    f'[serve] query it:  gls segment {self.model_path} "<prompt>" -O ply_overlay -o out.ply\n'
                    f"[serve] stop it:   gls stop {self.model_path}",
                    flush=True,
                )
            else:
                print(f"[serve] engine failed to start (see above / {log})", flush=True)
            return

        cfg = build_config(
            sfm=self.sfm or None,
            view_source=self.view_source,
            max_views=self.max_views,
            vram_budget_gb=self.vram_budget_gb,
            up=self.up,
            select=self.select,
            compete=(True if self.compete else None),
            fast_views=self.fast_views,
            device=self.device,
        )
        apply_recipe(cfg, recipe)
        t_start = time.time()
        engine = load_or_build_engine(self.model_path, cfg)  # one-time build

        if os.path.exists(sock_path):
            os.unlink(sock_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(sock_path)
        srv.listen(8)
        write_pidfile(sock_path)

        # Guaranteed cleanup: remove socket + pidfile however we exit (normal
        # shutdown, idle, signal, or crash). atexit covers the normal/crash paths;
        # the signal handlers convert SIGTERM/SIGINT/SIGHUP (what `gls stop`, a
        # bare `kill`, or a closing terminal send) into a clean break of the loop.
        def _cleanup() -> None:
            try:
                srv.close()
            finally:
                remove_runtime_files(sock_path)

        atexit.register(_cleanup)

        class _Stop(Exception):
            pass

        def _on_signal(signum, _frame):
            raise _Stop(signal.Signals(signum).name)

        for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            try:
                signal.signal(_sig, _on_signal)
            except (ValueError, OSError):
                pass

        keep_seconds = self.keep_alive * 60 if self.keep_alive and self.keep_alive > 0 else None
        idle_note = f"auto-stops after {self.keep_alive:g} min idle" if keep_seconds else "stays up until `gls stop`"
        last_activity = time.time()  # bumped ONLY by real queries, so polling status can't keep it alive
        print(
            f"[serve] engine warm ({engine.N:,} gaussians), listening ({idle_note}).\n"
            f'[serve] query it:  gls segment {self.model_path} "<prompt>" -O ply_overlay -o out.ply\n'
            f"[serve] check it:  gls status\n"
            f"[serve] stop it:   Ctrl-C  (or: gls stop {self.model_path})",
            flush=True,
        )
        try:
            while True:
                if keep_seconds is not None:
                    remaining = last_activity + keep_seconds - time.time()
                    if remaining <= 0:
                        print(f"[serve] idle for {self.keep_alive:g} min -> shutting down", flush=True)
                        break
                    srv.settimeout(min(remaining, 30.0))  # wake periodically to re-check the deadline
                else:
                    srv.settimeout(None)
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue  # not idle yet; loop re-checks the real deadline
                try:
                    req = recv_obj(conn)
                    cmd = req.get("cmd") or req.get("prompt")
                    if cmd in (":shutdown", "shutdown"):
                        send_obj(conn, {"ok": True, "msg": "shutting down"})
                        print("[serve] shutdown requested", flush=True)
                        break
                    if cmd in (":ping", "ping"):
                        send_obj(conn, {"ok": True, "N": engine.N})
                        continue  # liveness check: does NOT reset the idle timer
                    if cmd in (":status", "status"):
                        now = time.time()
                        idle_for = now - last_activity
                        send_obj(
                            conn,
                            {
                                "ok": True,
                                "pid": os.getpid(),
                                "model": str(self.model_path),
                                "N": engine.N,
                                "recipe": self.recipe,
                                "uptime_s": now - t_start,
                                "idle_s": idle_for,
                                "idle_remaining_s": (keep_seconds - idle_for) if keep_seconds else None,
                                "vram": engine.vram_report(),
                            },
                        )
                        continue  # status check: does NOT reset the idle timer
                    last_activity = time.time()
                    resp = handle_request(engine, req)
                    send_obj(conn, resp)
                    if resp.get("ok"):
                        t = resp.get("t")
                        ts = f"{t:.2f}s" if isinstance(t, (int, float)) else "?"
                        print(f'[serve] "{resp["prompt"]}" -> {resp["n"]:,}/{resp["N"]:,} ({ts})', flush=True)
                    else:
                        print(f"[serve] error: {resp.get('error')}", flush=True)
                except Exception as e:  # keep the daemon alive on a bad request
                    try:
                        send_obj(conn, {"ok": False, "error": str(e)})
                    except Exception:
                        pass
                    print(f"[serve] request failed: {e}", flush=True)
                finally:
                    conn.close()
        except (_Stop, KeyboardInterrupt) as e:
            reason = e.args[0] if getattr(e, "args", None) else "interrupt"
            print(f"\n[serve] {reason} -> stopping", flush=True)
        finally:
            _cleanup()
