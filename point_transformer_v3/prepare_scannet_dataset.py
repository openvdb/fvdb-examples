# SPDX-License-Identifier: Apache-2.0
"""
Modified from https://github.com/Pointcept/Pointcept.git

ScanNet Dataset Preparation Script

This script exports a subset of ScanNet dataset samples to JSON format for testing
and development purposes. It performs grid sampling to reduce point density and
ensures consistent point counts per sample.
"""

import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
        "pose",
    ]

    def __init__(
        self,
        split="train",
        data_root="data/dataset",
        ignore_index=-1,
    ):
        super(DefaultDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.ignore_index = ignore_index
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            split_list = [self.split]
        else:
            split_list = self.split

        data_list = []
        for split in split_list:
            if os.path.isfile(os.path.join(self.data_root, split)):
                with open(os.path.join(self.data_root, split)) as f:
                    data_list += [os.path.join(self.data_root, data) for data in json.load(f)]
            else:
                data_list += glob.glob(os.path.join(self.data_root, split, "*"))
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = os.path.basename(data_path)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        # Convert data types
        if "coord" in data_dict:
            data_dict["coord"] = data_dict["coord"].astype(np.float32)
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"].astype(np.float32)
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"].astype(np.float32)
        if "segment" in data_dict:
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        if "instance" in data_dict:
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)

        return data_dict

    def __getitem__(self, idx):
        return self.get_data(idx)

    def __len__(self):
        return len(self.data_list)


class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]

    def get_data(self, idx):
        data_dict = super().get_data(idx)

        if "segment20" in data_dict:
            data_dict["segment"] = data_dict.pop("segment20").reshape([-1]).astype(np.int32)
        elif "segment200" in data_dict:
            data_dict["segment"] = data_dict.pop("segment200").reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1

        return data_dict


def grid_sample_points(xyz: np.ndarray, color: np.ndarray, segment: np.ndarray, voxel_size: float = 0.1) -> tuple:
    """
    Perform grid sampling to downsample point cloud data by keeping one point per voxel.

    Divides 3D space into uniform voxels and selects one representative point per occupied
    voxel. Adds a corner point at the minimum coordinate for grid alignment.

    Args:
        xyz: Point coordinates (N, 3)
        color: Point colors (N, 3)
        segment: Point segment labels (N,)
        voxel_size: Grid cell size (default: 0.1)

    Returns:
        Tuple of (sampled_xyz, sampled_color, sampled_segment, sampled_grid_coords)
        Each array has M points where M <= N+1 due to downsampling and corner point addition.
    """
    xyz_min = xyz.min(0)
    grid_coords = (xyz - xyz_min) // voxel_size

    # Add corner point
    grid_coords = np.concatenate([np.zeros((1, 3)), grid_coords], axis=0)
    xyz = np.concatenate([xyz_min.reshape(1, 3), xyz], axis=0)
    color = np.concatenate([color.min(0).reshape(1, 3), color], axis=0)
    segment = np.concatenate([np.zeros((1,)), segment], axis=0)

    # Find unique grid cells
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)

    return xyz[unique_indices], color[unique_indices], segment[unique_indices], grid_coords[unique_indices]


def export_scannet_samples(
    data_root: str,
    output_file: str,
    num_samples: int = 10,
    split: str = "train",
    min_points: int = 2048,
    max_points: int = 4096,
    patch_size: int = 0,
    voxel_size: float = 0.1,
) -> str:
    """
    Export ScanNet samples to JSON format.

    Args:
        data_root: Root directory of ScanNet dataset
        output_file: Output JSON file path
        num_samples: Number of samples to export
        split: Dataset split to use
        min_points: Minimum points per sample
        max_points: Maximum points per sample
        patch_size: the exported point cloud should contain multiple of patch_size points
        voxel_size: voxel size for grid sampling
    Returns:
        Path to exported file
    """
    # Setup logging to print to terminal
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    logger = logging.getLogger(__name__)

    # Collect scene paths
    split_path = Path(data_root) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")

    scene_paths = list(split_path.glob("*"))
    logger.info(f"Found {len(scene_paths)} scenes in {split} split")

    if len(scene_paths) < num_samples:
        logger.warning(f"Requested {num_samples} samples but only {len(scene_paths)} available")
        num_samples = len(scene_paths)

    # Randomly sample scenes
    np.random.seed(42)
    # create a permutation of the scene paths
    selected_paths = np.random.permutation(scene_paths)

    # Initialize dataset
    dataset = ScanNetDataset(data_root=data_root, split=split)

    exported_samples = []
    export_count = 0

    for i, scene_path in enumerate(selected_paths):
        if export_count >= num_samples:
            break

        scene_name = scene_path.name
        logger.info(f"Processing {i+1}/{len(selected_paths)}: {scene_name}")

        # Find scene index in dataset
        scene_index = None
        for idx, data_path in enumerate(dataset.data_list):
            if scene_name in data_path:
                scene_index = idx
                break

        if scene_index is None:
            logger.warning(f"Scene {scene_name} not found in dataset")
            continue

        try:
            # Load and process data
            data_dict = dataset.get_data(scene_index)
            xyz, color, segment, grid_coords = grid_sample_points(
                data_dict["coord"], data_dict["color"], data_dict["segment"], voxel_size=voxel_size
            )

            # Check point count constraints
            if not (min_points <= len(xyz) <= max_points):
                continue

            if patch_size > 0:
                # pad the point cloud to the nearest multiple of patch_size
                xyz = xyz[: len(xyz) // patch_size * patch_size]
                color = color[: len(color) // patch_size * patch_size]
                segment = segment[: len(segment) // patch_size * patch_size]
                grid_coords = grid_coords[: len(grid_coords) // patch_size * patch_size]

            # Prepare sample data
            sample = {
                "num_points": len(xyz),
                "xyz": xyz.tolist(),
                "grid_coords": grid_coords.tolist(),
                "color": color.tolist(),
                "label": segment.tolist(),
            }

            exported_samples.append(sample)
            export_count += 1
            logger.info(f"Exported {scene_name} with {len(xyz)} points")

        except Exception as e:
            logger.error(f"Failed to process {scene_name}: {e}")
            continue

    # Save to JSON
    with open(output_file, "w") as f:
        # json.dump(exported_samples, f, indent=2)
        json.dump(exported_samples, f)  # no indent

    # Log summary
    total_points = sum(s["num_points"] for s in exported_samples)
    avg_points = total_points / len(exported_samples) if exported_samples else 0

    logger.info(
        f"Export complete: {len(exported_samples)} samples, "
        f"{total_points} total points, {avg_points:.1f} avg points/sample"
    )

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Export ScanNet dataset samples")
    parser.add_argument("--data-root", required=True, help="ScanNet dataset root directory")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to export")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--min-points", type=int, default=50000, help="Minimum points per sample")
    parser.add_argument("--max-points", type=int, default=100000, help="Maximum points per sample")
    parser.add_argument("--patch-size", type=int, default=1024, help="Maximum points per sample")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for grid sampling")

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Export samples
    exported_file = export_scannet_samples(
        data_root=args.data_root,
        output_file=args.output,
        num_samples=args.num_samples,
        split=args.split,
        min_points=args.min_points,
        max_points=args.max_points,
        patch_size=args.patch_size,
        voxel_size=args.voxel_size,
    )

    print(f"Export completed: {exported_file}")


if __name__ == "__main__":
    main()

# Create scannet_samples_small.json
# python prepare_scannet_dataset.py --data-root /home/hexuz/openvdb/fvdb/projects/sparse_attention/Pointcept/data/scannet --output data/scannet_samples_small.json --num-samples 8 --split train --min-points 2048 --max-points 4096 --voxel-size 0.1 --patch-size 1024

# Create scannet_samples_large.json
# python prepare_scannet_dataset.py --data-root /home/hexuz/openvdb/fvdb/projects/sparse_attention/Pointcept/data/scannet --output data/scannet_samples_large.json --num-samples 4 --split train --min-points 50000 --max-points 100000 --voxel-size 0.02 --patch-size 1024

