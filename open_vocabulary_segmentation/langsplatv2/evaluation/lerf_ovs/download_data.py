# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""CLI script to download the LERF-OVS evaluation dataset."""
from dataclasses import dataclass
from pathlib import Path

import tyro
from langsplatv2.evaluation.datasets import set_dataset_root
from langsplatv2.evaluation.datasets.lerf import download_lerf_data


@dataclass
class DownloadLERFData:
    """Download the LERF-OVS dataset for open-vocabulary segmentation evaluation."""

    dataset_root: Path = Path("data")
    """Root directory to store downloaded datasets."""

    def main(self):
        set_dataset_root(self.dataset_root)
        download_lerf_data()


if __name__ == "__main__":
    tyro.cli(DownloadLERFData).main()
