# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import tyro

from geolangsplat.cli import (
    Bake,
    Check,
    Doctor,
    Explore,
    Render,
    Segment,
    Serve,
    Show,
    Status,
    Stop,
)

# Mirror the real `gls()` union so the test fails if a command is dropped from dispatch.
_UNION = Segment | Bake | Check | Doctor | Serve | Show | Explore | Render | Status | Stop


def test_segment_minimal_parses():
    s = tyro.cli(_UNION, args=["segment", "m.ply", "house", "-r", "satellite"])
    assert type(s).__name__ == "Segment"
    assert s.prompt == "house" and s.recipe == "satellite"


def test_segment_output_parses():
    s = tyro.cli(_UNION, args=["segment", "m.ply", "road", "-O", "ply_overlay", "-o", "r.ply", "-t", "0.4"])
    assert type(s).__name__ == "Segment"
    assert s.output == "ply_overlay" and s.select == 0.4
    assert not hasattr(s, "warm")  # engine lifecycle lives in serve/stop, not segment


def test_serve_background_and_keep_alive_parses():
    s = tyro.cli(_UNION, args=["serve", "m.ply", "-r", "satellite", "-b", "--keep-alive", "15"])
    assert type(s).__name__ == "Serve"
    assert s.recipe == "satellite" and s.background is True and s.keep_alive == 15.0


def test_serve_keep_alive_pin_parses():
    s = tyro.cli(_UNION, args=["serve", "m.ply", "--keep-alive", "0"])
    assert type(s).__name__ == "Serve" and s.keep_alive == 0.0


def test_serve_socket_path_parses():  # the warm-spawn path passes this through
    s = tyro.cli(_UNION, args=["serve", "m.ply", "--socket-path", "/tmp/x.sock"])
    assert type(s).__name__ == "Serve" and s.socket_path == "/tmp/x.sock"


def test_show_parses():
    s = tyro.cli(_UNION, args=["show", "out.ply", "-p", "8090"])
    assert type(s).__name__ == "Show"
    assert str(s.model_path) == "out.ply" and s.viewer_port == 8090


def test_stop_parses():
    s = tyro.cli(_UNION, args=["stop", "m.ply"])
    assert type(s).__name__ == "Stop"
    assert str(s.model_path) == "m.ply"


def test_stop_all_parses():  # no model -> stop everything
    s = tyro.cli(_UNION, args=["stop"])
    assert type(s).__name__ == "Stop"
    assert s.model_path is None


def test_status_parses():
    s = tyro.cli(_UNION, args=["status"])
    assert type(s).__name__ == "Status"


def test_segment_low_vram_on_by_default():
    s = tyro.cli(_UNION, args=["segment", "m.ply", "house"])
    assert s.low_vram is True  # one-shot defaults to bounded VRAM
    assert s.stream_early_stop == "auto"  # capture-gated unless forced
    assert s.stream_chunk is None  # falls through to the config default


def test_segment_streaming_flags_parse():
    s = tyro.cli(
        _UNION,
        args=[
            "segment",
            "m.ply",
            "tree",
            "--no-low-vram",
            "--stream-early-stop",
            "off",
            "--stream-chunk",
            "16",
        ],
    )
    assert s.low_vram is False
    assert s.stream_early_stop == "off" and s.stream_chunk == 16


def test_segment_streaming_flags_wire_into_config():
    # build_config must forward the streaming knobs the one-shot path reads.
    from geolangsplat.cli._common import build_config

    cfg = build_config(low_vram=True, stream_early_stop="on", stream_chunk=4)
    assert cfg.low_vram is True
    assert cfg.stream_early_stop == "on" and cfg.stream_chunk == 4


def test_doctor_render_explore_parse():
    assert type(tyro.cli(_UNION, args=["doctor"])).__name__ == "Doctor"
    assert type(tyro.cli(_UNION, args=["render", "m.ply", "-o", "x.png"])).__name__ == "Render"
    assert type(tyro.cli(_UNION, args=["explore", "m.ply"])).__name__ == "Explore"
