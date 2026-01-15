from pathlib import Path

from . import get_dataset_root
from .util import download_dropbox_folder, download_huggingface_dataset

nvos_url = "https://www.dropbox.com/sh/sdgr4mewkhjsg00/AACIKecIwzCHCGma5kkKyLTpa?dl=0"
modified_llff_hf_repo = "jswartz/nex-real-forward-facing"


def get_nvos_data_path() -> Path:
    dataset_root = get_dataset_root()
    if dataset_root is None:
        raise ValueError("Dataset root is not set")
    result = dataset_root / "nvos"
    if not result.exists():
        result.mkdir(parents=True, exist_ok=True)
    return result


def download_nvos_data():
    """
    Download the NVOS dataset from Dropbox and HuggingFace.

    Downloads:
        - NVOS labels from Dropbox
        - Modified LLFF scenes from HuggingFace (jswartz/nex-real-forward-facing)

    The data will be saved to the NVOS data path (dataset_root/nvos/).
    """
    output_path = get_nvos_data_path()

    print(f"Downloading NVOS dataset to: {output_path}")

    # Download labels from Dropbox
    labels_dir = output_path / "labels"
    if not labels_dir.exists() or not any(labels_dir.iterdir()):
        labels_dir.mkdir(parents=True, exist_ok=True)
        download_dropbox_folder(nvos_url, labels_dir)
        print(f"Labels downloaded to: {labels_dir}")
    else:
        print(f"Labels already exist at: {labels_dir}")

    # Download scenes from HuggingFace
    scenes_dir = output_path / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    download_huggingface_dataset(modified_llff_hf_repo, scenes_dir)

    print(f"\nNVOS dataset download complete: {output_path}")
