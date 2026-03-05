#!/usr/bin/env python3
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Train all 3 LangSplatV2 feature levels for a single scene.

Wraps ``train_langsplatv2.py``, running it once per feature level and
collecting the final checkpoints into a results directory ready for
evaluation.

Any arguments not recognised by this wrapper are forwarded verbatim to
``train_langsplatv2.py`` (e.g. ``--config.max-steps``,
``--preprocess.sam-model``, ``--dataset-type``).

Usage:
    # Minimal -- trains levels 1, 2, 3 with default settings:
    python scripts/train_all_levels.py \\
        --sfm-dataset-path /path/to/colmap/scene \\
        --reconstruction-path /path/to/scene.ply

    # With extra training options:
    python scripts/train_all_levels.py \\
        --sfm-dataset-path /data/my_scene \\
        --reconstruction-path /data/my_scene.ply \\
        --name my_scene \\
        --results-dir my_results \\
        --config.max-steps 10000 \\
        --preprocess.sam-model sam1
"""
import argparse
import pathlib
import shutil
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_langsplatv2.py"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LangSplatV2 at all 3 feature levels and collect checkpoints.",
        epilog=(
            "Unrecognised arguments are forwarded to train_langsplatv2.py. "
            "For example: --config.max-steps 10000 --preprocess.sam-model sam1"
        ),
    )
    parser.add_argument(
        "--sfm-dataset-path",
        type=pathlib.Path,
        required=True,
        help="Path to the SfM dataset (COLMAP, simple_directory, or E57).",
    )
    parser.add_argument(
        "--reconstruction-path",
        type=pathlib.Path,
        required=True,
        help="Path to the pre-trained Gaussian splat reconstruction (.ply or .pt).",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=pathlib.Path("langsplatv2_results"),
        help="Directory to collect final checkpoints into (default: langsplatv2_results).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Scene name used for run directories and checkpoint files. "
        "Defaults to the reconstruction file stem.",
    )
    parser.add_argument(
        "--log-path",
        type=pathlib.Path,
        default=pathlib.Path("langsplatv2_logs"),
        help="Directory for per-run training logs and checkpoints (default: langsplatv2_logs).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Feature levels to train (default: 1 2 3).",
    )

    args, extra = parser.parse_known_args()
    name = args.name or args.reconstruction_path.stem

    failed_levels: list[int] = []

    total_levels = len(args.levels)
    for idx, level in enumerate(args.levels, 1):
        run_name = f"{name}_level_{level}"
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--sfm-dataset-path", str(args.sfm_dataset_path),
            "--reconstruction-path", str(args.reconstruction_path),
            "--config.feature-level", str(level),
            "--run-name", run_name,
            "--log-path", str(args.log_path),
            *extra,
        ]

        print(f"\n{'=' * 60}")
        print(f"  Training level {level} ({idx}/{total_levels}): {run_name}")
        print(f"{'=' * 60}\n")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nERROR: Training failed for level {level} (exit code {result.returncode})")
            failed_levels.append(level)

    # -- Collect checkpoints ------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    collected = 0

    for level in args.levels:
        if level in failed_levels:
            continue
        run_name = f"{name}_level_{level}"
        src = args.log_path / run_name / "final_checkpoint.pt"
        dst = args.results_dir / f"{run_name}.pt"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Collected: {src} -> {dst}")
            collected += 1
        else:
            print(f"  WARNING: {src} not found, skipping")

    # -- Summary ------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"  Done -- {collected}/{len(args.levels)} checkpoints in {args.results_dir}")
    if failed_levels:
        print(f"  Failed levels: {failed_levels}")
    print(f"{'=' * 60}")

    if failed_levels:
        sys.exit(1)


if __name__ == "__main__":
    main()
