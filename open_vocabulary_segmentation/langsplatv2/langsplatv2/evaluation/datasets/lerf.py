# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path

from . import get_dataset_root
from .util import download_google_drive_file

# Google Drive file ID extracted from the download link in the LangSplatV2 README:
# https://drive.google.com/file/d/1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt/view?usp=sharing
LERF_GDRIVE_FILE_ID = "1QF1Po5p5DwTjFHu6tnTeYs_G0egMVmHt"

# Scenes available in the LERF-OVS dataset
LERF_SCENES = [
    "figurines",
    "ramen",
    "teatime",
    "waldo_kitchen",
]


def get_lerf_data_path() -> Path:
    """Get the path to the LERF-OVS dataset root.

    Returns:
        Path to the ``lerf_ovs`` directory inside the dataset root.

    Raises:
        ValueError: If the dataset root has not been set via ``set_dataset_root()``.
    """
    dataset_root = get_dataset_root()
    if dataset_root is None:
        raise ValueError("Dataset root is not set. Call set_dataset_root() first.")
    result = dataset_root / "lerf_ovs"
    if not result.exists():
        result.mkdir(parents=True, exist_ok=True)
    return result


def download_lerf_data():
    """
    Download the LERF-OVS dataset from Google Drive.

    The dataset contains COLMAP scenes and labelme-format ground truth labels
    for open-vocabulary segmentation evaluation.

    Expected layout after download::

        lerf_ovs/
            label/
                <scene_name>/
                    frame_XXXXX.json
                    frame_XXXXX.jpg
            <scene_name>/        (COLMAP scene)
                images/
                sparse/
                output/

    The data will be saved to the LERF data path (dataset_root/lerf_ovs/).
    """
    output_path = get_lerf_data_path()

    print(f"Downloading LERF-OVS dataset to: {output_path}")

    # Check if data already exists
    label_dir = output_path / "label"
    if label_dir.exists() and any(label_dir.iterdir()):
        print(f"LERF-OVS data already exists at: {output_path}")
        print("Delete the directory to re-download.")
        return

    download_google_drive_file(
        file_id=LERF_GDRIVE_FILE_ID,
        output_path=output_path.parent,
        filename="lerf_ovs.zip",
    )

    print(f"\nLERF-OVS dataset download complete: {output_path}")
