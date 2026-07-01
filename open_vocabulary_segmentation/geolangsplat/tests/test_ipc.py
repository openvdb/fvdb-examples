# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os
import socket

import geolangsplat.outputs as _out
from geolangsplat.ipc import (
    cleanup_stale,
    daemon_alive,
    default_socket,
    handle_request,
    kill_daemon,
    pid_alive,
    pidfile_for,
    read_pid,
    recv_obj,
    remove_runtime_files,
    send_obj,
    write_pidfile,
)


class _FakeGS:
    @classmethod
    def from_tensors(cls, **kw):
        inst = cls()
        inst.kw = kw
        return inst

    def save_ply(self, path):
        with open(path, "w") as f:
            f.write(f"n={self.kw['means'].shape[0]}\n")


def test_default_socket_stable_and_per_model():
    a = default_socket("model.ply")
    assert a == default_socket("model.ply")  # stable
    assert a != default_socket("other.ply")  # per-model
    assert a.endswith(".sock")


def test_send_recv_roundtrip():
    a, b = socket.socketpair()
    send_obj(a, {"prompt": "house", "n": 3})
    got = recv_obj(b)
    assert got == {"prompt": "house", "n": 3}
    a.close()
    b.close()


def test_handle_request_mask(fake_engine):
    resp = handle_request(fake_engine, {"prompt": "house"})
    assert resp["ok"] is True
    assert resp["N"] == fake_engine.N
    assert 0 <= resp["n"] <= fake_engine.N
    assert resp["path"] is None


def test_handle_request_empty_prompt(fake_engine):
    resp = handle_request(fake_engine, {"prompt": "   "})
    assert resp["ok"] is False


def test_handle_request_output_needs_path(fake_engine):
    resp = handle_request(fake_engine, {"prompt": "house", "output": "ply_overlay"})
    assert resp["ok"] is False


def test_handle_request_writes_overlay(fake_engine, tmp_path, monkeypatch):
    monkeypatch.setattr(_out, "_gs", lambda: _FakeGS)
    out = tmp_path / "h.ply"
    resp = handle_request(fake_engine, {"prompt": "house", "output": "ply_overlay", "out_path": str(out)})
    assert resp["ok"] is True
    assert resp["path"] == str(out)
    assert out.exists()


def test_handle_request_applies_select_override(fake_engine):
    handle_request(fake_engine, {"prompt": "house", "select": 0.99})
    assert fake_engine.cfg.select == 0.99


def test_daemon_alive_false_when_nothing_listening(tmp_path):
    assert daemon_alive(str(tmp_path / "nope.sock"), timeout=0.5) is False


def test_pid_alive_self_and_bogus():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_147_480_000) is False  # almost certainly unused PID


def test_pidfile_roundtrip_and_remove(tmp_path):
    sock = str(tmp_path / "x.sock")
    write_pidfile(sock, 4242)
    assert os.path.exists(pidfile_for(sock))
    assert read_pid(sock) == 4242
    remove_runtime_files(sock)
    assert not os.path.exists(pidfile_for(sock))
    assert read_pid(sock) is None


def test_kill_daemon_not_running_is_clean(tmp_path):
    sock = str(tmp_path / "dead.sock")
    write_pidfile(sock, 2_147_480_000)  # stale pid, no process, no socket bound
    assert kill_daemon(sock, grace=0.2) == "not running"
    assert not os.path.exists(pidfile_for(sock))


def test_cleanup_stale_removes_when_pid_dead(tmp_path):
    sock = str(tmp_path / "stale.sock")
    open(sock, "w").close()  # leftover socket file, no listener
    write_pidfile(sock, 2_147_480_000)  # dead pid
    assert cleanup_stale(sock) is True
    assert not os.path.exists(sock)
    assert read_pid(sock) is None
