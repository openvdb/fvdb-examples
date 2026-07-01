# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Shared CLI helpers."""
from __future__ import annotations

import pathlib
import sys

from ..config import GeoLangSplatConfig

# BaseCommand: subclass fvdb-reality-capture's so these commands can drop straight
# into `frgs` -- but ONLY when frc is already imported (i.e. we're running inside
# frgs, where torch/fvdb are loaded anyway). For the standalone `gls` CLI we use a
# local stand-in with the identical contract, so importing the CLI stays torch-free
# and the lightweight paths (daemon-attach segment, status, stop) start instantly.
BaseCommand = None
if "fvdb_reality_capture" in sys.modules:  # pragma: no cover - frc-present path
    try:
        from fvdb_reality_capture.cli import BaseCommand
    except Exception:
        BaseCommand = None

if BaseCommand is None:
    from abc import ABC, abstractmethod

    class BaseCommand(ABC):  # type: ignore[no-redef]
        """Local stand-in matching fvdb_reality_capture.cli.BaseCommand."""

        @abstractmethod
        def execute(self) -> None: ...


def build_config(**overrides) -> GeoLangSplatConfig:
    """Build a config, applying only the overrides the user actually provided
    (``None`` values are ignored so recipe presets can still fill them)."""
    cfg = GeoLangSplatConfig()
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    return cfg


def resolve_recipe(recipe, view_source=None):
    """Default the recipe when the user did not pick one.

    Falls back to ``"aerial"`` when the user asked for ground-truth photos
    (``--view-source images``), since that path wants the curated photo recipe;
    otherwise ``"auto"`` (geometry-driven views), so every command frames a scene
    sensibly out of the box.
    """
    if recipe:
        return recipe
    if view_source == "images":
        return "aerial"
    return "auto"


def read_vocab_file(path) -> list[str]:
    """Read a vocabulary from a text file: one word/phrase per line, '#' comments."""
    text = pathlib.Path(path).read_text()
    words: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            words.append(line)
    return words
