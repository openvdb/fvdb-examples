from dataclasses import dataclass
from pathlib import Path

import tyro
from garfvdb.evaluation.datasets import set_dataset_root
from garfvdb.evaluation.datasets.nvos import download_nvos_data


@dataclass
class DownloadNVOSData:
    dataset_root: Path = Path("data")

    def main(self):
        set_dataset_root(self.dataset_root)
        download_nvos_data()


if __name__ == "__main__":
    tyro.cli(DownloadNVOSData).main()
