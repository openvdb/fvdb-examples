# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests


def download_example_data(file_name: str, logger: logging.Logger):
    """
    Download ScanNet samples JSON from the fvdb-test-data repository.
    """
    raw_url = f"https://raw.githubusercontent.com/voxel-foundation/fvdb-test-data/scannet/unit_tests/ptv3/{file_name}"

    # Script is in scripts/data/, so go up one level to get project root
    project_root = Path(__file__).parent.parent.parent.resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / file_name

    try:
        logger.info(f"Downloading ScanNet samples from: {raw_url}")
        response = requests.get(raw_url)
        response.raise_for_status()  # Raise an exception for bad status codes

        json_data = response.json()

        with open(output_file, "w") as f:
            json.dump(json_data, f, indent=2)

        logger.info(f"Successfully downloaded {len(json_data)} samples to: {output_file}")
        return str(output_file)

    except requests.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    file_names = [
        "scannet_samples_small.json",
        "scannet_samples_large.json",
        "scannet_samples_small_output_gt.json",
        "scannet_samples_large_output_gt.json",
    ]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    for file_name in file_names:
        download_example_data(file_name, logger)
