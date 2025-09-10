# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
import os

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class Sun3dDataset(Dataset):
    """
    Sun3d dataset for testing and evaluation.

    Sun3D is a dataset of RGB-D images with camera poses.
    Each scene comprises a sequence of images, depth and pose files captured from a single camera.
    The colors, depths, and poses in the sequence are named as:
     - <frame_name>.color.png
     - <frame_name>.depth.png
     - <frame_name>.pose.txt
    The camera intrinsics (projection matrix) is stored in a separate file:
     - camera-intrinsics.txt

    Attributes:
        root (str): Root directory of the dataset.
        camera_projection_matrix (torch.Tensor): Camera projection matrix loaded from the intrinsics file.
        sequence_name (str): Name of the sequence directory.
        seq_dir (str): Full path to the sequence directory.
    """

    def __init__(self, dataset_path: str) -> None:
        """
        Creates a new `Sun3dDataset` instance from a path to the data.

        You can dowload the Sun3D dataset from https://vision.princeton.edu/projects/2016/3DMatch/
        or by running the script `download_example_data.py` in this directory to download a single
        RGBD sequence for testing.

        Args:
            dataset_path (str): Path to the root directory of the Sun3D dataset.
        """
        super().__init__()
        self._dataset_path: str = dataset_path
        if not os.path.exists(self._dataset_path):
            raise FileNotFoundError(f"Dataset path {self._dataset_path} does not exist.")

        projection_matrix: np.ndarray = np.loadtxt(os.path.join(self._dataset_path, "camera-intrinsics.txt"))
        self._projection_matrix: torch.Tensor = torch.from_numpy(projection_matrix).float()
        self._sequence_name: str = list(
            sorted([d for d in os.listdir(self._dataset_path) if os.path.isdir(os.path.join(self._dataset_path, d))])
        )[0]
        self._sequence_path: str = os.path.join(self._dataset_path, self._sequence_name)

        self._frame_names: list[str] = list(sorted({f.split(".")[0] for f in os.listdir(self._sequence_path)}))

    def __len__(self) -> int:
        """
        Returns the number of frames in the dataset.

        Returns:
            int: Number of frames in the dataset.
        """
        return len(self._frame_names)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Get the i^th frame of the dataset.
        A frame consists of:
            - rgb: RGB image tensor of shape HxWx3
            - depth: Depth image tensor of shape HxWx1
            - cam_to_world_matrix: The 4x4 transformation matrix from camera to world coordinates.
            - projection_matrix: The 3x3 camera projection matrix

        Args:
            index (int): Index of the frame to retrieve. Must be in the range [0, len(self)).

        Returns:
            dict[str, torch.Tensor]: A dictionary containing:
                - "rgb": RGB image tensor of shape HxWx3
                - "depth": Depth image tensor of shape HxWx1
                - "cam_to_world_matrix": The 4x4 transformation matrix from camera to world coordinates.
                - "projection_matrix": The 3x3 camera projection matrix.
        """

        # Load the raw data
        frame_name = self._frame_names[index]
        rgb_np = cv2.imread(os.path.join(self._sequence_path, f"{frame_name}.color.png"))
        depth_np = cv2.imread(os.path.join(self._sequence_path, f"{frame_name}.depth.png"), cv2.IMREAD_UNCHANGED)
        pose_np = np.loadtxt(os.path.join(self._sequence_path, f"{frame_name}.pose.txt"))

        # Color
        rgb = torch.from_numpy(rgb_np)
        assert rgb.dtype == torch.uint8, "Only 8-bit RGB images supported"
        assert rgb.shape[-1] == 3, "Only 3-channel RGB images supported by nvblox"

        # Depth
        depth = torch.from_numpy(depth_np.astype(np.float32) / 1000.0).to(torch.float32)
        depth = depth.squeeze()
        depth = depth.unsqueeze(dim=-1)
        assert depth.shape[-1] == 1, "Only 1-channel depth images supported by nvblox"

        # Pose
        cam_to_world_matrix = torch.from_numpy(pose_np).to(torch.float32)

        # Post-conditions

        return {
            "rgb": rgb,
            "depth": depth,
            "cam_to_world_matrix": cam_to_world_matrix,
            "projection_matrix": self._projection_matrix,
        }
