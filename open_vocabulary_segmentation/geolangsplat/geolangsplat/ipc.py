# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Tiny line-delimited JSON IPC over a Unix domain socket.

Backs the warm-engine daemon: ``gls serve`` holds the caches in one process, and
``gls segment`` auto-detects that process and queries it. This is *only* an
optimization -- the daemon is a thin wrapper around the same stateless
:func:`geolangsplat.api.segment`, so nothing about the public API depends on it.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pathlib
import socket
import time

SOCKET_GLOB = "/tmp/gls-*.sock"

# Single lifecycle knob's default: a warm engine auto-stops after this many idle
# minutes so a forgotten daemon never pins the GPU. Sliding window; reset only by
# real queries. 0/negative = pin until explicitly stopped.
DEFAULT_KEEP_ALIVE_MIN = 15.0


def default_socket(model_path) -> str:
    """A stable per-model socket path under /tmp, so client and server agree
    without the user having to pass an explicit ``--socket``."""
    key = str(pathlib.Path(model_path).expanduser().resolve())
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return f"/tmp/gls-{h}.sock"


def list_sockets() -> list[str]:
    """All candidate GeoLangSplat daemon sockets on this machine."""
    return sorted(glob.glob(SOCKET_GLOB))


# -- pidfile: lets `gls stop` hard-kill a wedged daemon and lets cleanup detect a
# dead one even when its stale socket file lingers. Written next to the socket. --


def pidfile_for(socket_path: str) -> str:
    return socket_path + ".pid"


def write_pidfile(socket_path: str, pid: int | None = None) -> None:
    try:
        with open(pidfile_for(socket_path), "w") as f:
            f.write(str(pid if pid is not None else os.getpid()))
    except OSError:
        pass


def read_pid(socket_path: str) -> int | None:
    try:
        with open(pidfile_for(socket_path)) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists (signal 0 probes without killing)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def remove_runtime_files(socket_path: str) -> None:
    """Remove the socket + pidfile for a daemon (idempotent)."""
    for p in (socket_path, pidfile_for(socket_path)):
        try:
            os.unlink(p)
        except OSError:
            pass


def cleanup_stale(socket_path: str) -> bool:
    """Remove the socket + pidfile of a daemon that is gone. Returns True if a
    stale file was removed. Leaves a live daemon untouched."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(socket_path)
        return False  # someone is listening -> not stale
    except ConnectionRefusedError:
        remove_runtime_files(socket_path)  # bound earlier, nobody home now
        return True
    except OSError:
        # missing socket or connect timeout: trust the pidfile to decide.
        pid = read_pid(socket_path)
        if pid is not None and not pid_alive(pid):
            remove_runtime_files(socket_path)
            return True
        return False
    finally:
        s.close()


def kill_daemon(socket_path: str, grace: float = 5.0) -> str:
    """Guaranteed-best-effort shutdown: ask politely over IPC, then SIGTERM, then
    SIGKILL, always cleaning up the socket + pidfile. Returns a short status word
    ('not running' | 'stopped' | 'killed' | 'stop attempted')."""
    import signal as _signal

    pid = read_pid(socket_path)
    alive = daemon_alive(socket_path)
    if not alive and (pid is None or not pid_alive(pid)):
        remove_runtime_files(socket_path)
        return "not running"

    if alive:  # 1) polite, lets the daemon free VRAM on its own terms
        try:
            request(socket_path, {"cmd": ":shutdown"}, timeout=grace)
        except OSError:
            pass
        deadline = time.time() + grace
        while time.time() < deadline:
            if not daemon_alive(socket_path, timeout=0.5) and not (pid and pid_alive(pid)):
                remove_runtime_files(socket_path)
                return "stopped"
            time.sleep(0.2)

    if pid and pid_alive(pid):  # 2) escalate: it's wedged or won't exit
        for sig in (_signal.SIGTERM, _signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except OSError:
                break
            deadline = time.time() + grace
            while time.time() < deadline and pid_alive(pid):
                time.sleep(0.2)
            if not pid_alive(pid):
                break

    remove_runtime_files(socket_path)
    return "killed" if (pid and not pid_alive(pid)) else "stopped"


def request_status(socket_path: str, timeout: float = 2.0) -> dict | None:
    """Ask a daemon for its status payload, or None if it isn't answering."""
    try:
        resp = request(socket_path, {"cmd": ":status"}, timeout=timeout)
    except OSError:
        return None
    return resp if resp.get("ok") else None


def send_obj(conn: socket.socket, obj: dict) -> None:
    conn.sendall((json.dumps(obj) + "\n").encode())


def recv_obj(conn: socket.socket) -> dict:
    """Read one newline-terminated JSON object. Returns {} on empty/closed."""
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
    line = buf.decode().strip()
    return json.loads(line) if line else {}


def request(socket_path: str, obj: dict, timeout: float | None = 600.0) -> dict:
    """Connect to a running daemon, send one request, return its reply."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout is not None:
        s.settimeout(timeout)
    try:
        s.connect(socket_path)
        send_obj(s, obj)
        return recv_obj(s)
    finally:
        s.close()


def daemon_alive(socket_path: str, timeout: float = 2.0) -> bool:
    """True if a daemon is listening on ``socket_path`` and answers a ping."""
    try:
        resp = request(socket_path, {"cmd": ":ping"}, timeout=timeout)
    except OSError:  # ConnectionRefused / FileNotFound / timeout all subclass OSError
        return False
    return bool(resp.get("ok"))


def spawn_daemon(
    model_path,
    *,
    recipe: str | None = None,
    sfm: str | None = None,
    view_source: str | None = None,
    max_views: int | None = None,
    vram_budget_gb: float | None = None,
    up: str | None = None,
    select: float | None = None,
    compete: bool = False,
    fast_views: int | None = None,
    device: str | None = None,
    socket_path: str | None = None,
    log_path: str | None = None,
    keep_alive: float | None = None,
):
    """Launch ``gls serve`` as a detached background process.

    Returns ``(proc, log_path)``; ``proc.poll()`` lets the caller notice a crash
    instead of waiting out the whole readiness timeout.
    """
    import subprocess
    import sys

    sock = socket_path or default_socket(model_path)
    log = log_path or (sock + ".log")
    open(log, "wb").close()  # truncate previous log so we stream only this run
    args = [sys.executable, "-m", "geolangsplat", "serve", str(model_path), "--socket-path", sock]
    if recipe:
        args += ["--recipe", recipe]
    if sfm:
        args += ["--sfm", sfm]
    if view_source:
        args += ["--view-source", view_source]
    if max_views is not None:
        args += ["--max-views", str(max_views)]
    if vram_budget_gb is not None:
        args += ["--vram-budget-gb", str(vram_budget_gb)]
    if up:
        args += ["--up", up]
    if select is not None:
        args += ["--select", str(select)]
    if compete:
        args += ["--compete"]
    if fast_views is not None:
        args += ["--fast-views", str(fast_views)]
    if keep_alive is not None:
        args += ["--keep-alive", str(keep_alive)]
    if device:
        args += ["--device", device]
    fh = open(log, "ab")
    proc = subprocess.Popen(
        args,
        stdout=fh,
        stderr=fh,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach: survives this CLI process exiting
    )
    return proc, log


def wait_ready(
    socket_path: str,
    proc=None,
    log_path: str | None = None,
    timeout: float = 1800.0,
    interval: float = 0.5,
    stream: bool = True,
) -> bool:
    """Wait until the daemon answers a ping. Fails fast if ``proc`` exits first,
    and (when ``stream``) tails the build log so the wait does not look hung."""
    import time

    t0 = time.time()
    pos = 0
    while time.time() - t0 < timeout:
        if stream and log_path:
            pos = _tail(log_path, pos)
        if daemon_alive(socket_path, timeout=2.0):
            if stream and log_path:
                _tail(log_path, pos)
            return True
        if proc is not None and proc.poll() is not None:  # daemon died during build
            if stream and log_path:
                _tail(log_path, pos)
            return False
        time.sleep(interval)
    return False


def _tail(log_path: str, pos: int) -> int:
    """Print new bytes appended to ``log_path`` since ``pos``; return new offset."""
    try:
        with open(log_path, "r", errors="replace") as fh:
            fh.seek(pos)
            chunk = fh.read()
            if chunk:
                print(chunk, end="", flush=True)
            return fh.tell()
    except OSError:
        return pos


def handle_request(engine, req: dict) -> dict:
    """Run one query request against a prebuilt engine and (optionally) write a file.

    Reuses :func:`geolangsplat.api.segment` so output formats stay identical to
    the one-shot CLI. Per-request ``select``/``compete`` overrides persist on the
    engine config (a session naturally remembers your last setting).
    """
    from .api import segment

    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}

    if req.get("select") is not None:
        engine.cfg.select = float(req["select"])
    if req.get("margin") is not None:
        engine.cfg.margin = float(req["margin"])
    if req.get("compete") is not None:
        engine.cfg.compete = bool(req["compete"])

    output = req.get("output") or "mask"
    out_path = req.get("out_path")
    if output != "mask" and not out_path:
        return {"ok": False, "error": f"output={output!r} requires out_path"}

    res = segment(engine.model, prompt, engine=engine, output=output, out_path=out_path)
    return {
        "ok": True,
        "prompt": prompt,
        "n": res.num_selected,
        "N": int(res.scores.shape[0]),
        "t": res.stats.get("query_seconds"),
        "output": output,
        "path": (str(out_path) if output != "mask" else None),
    }
