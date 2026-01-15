from pathlib import Path
from zipfile import ZipFile

import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_dropbox_folder(url: str, output_path: Path) -> Path:
    """
    Download a shared Dropbox folder as a zip file.

    Args:
        url: Dropbox shared folder URL
        output_path: Directory to save and extract to

    Returns:
        Path to the extracted folder
    """
    # Convert shared link to direct download format
    direct_url = url.replace("dl=0", "dl=1")

    zip_path = output_path / "nvos_dropbox.zip"

    print("Downloading NVOS data from Dropbox...")

    # Use headers to avoid being blocked and handle redirects
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    response = requests.get(direct_url, stream=True, headers=headers, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Dropbox") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print("Extracting Dropbox archive...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(output_path)

    # Clean up zip file
    zip_path.unlink()

    return output_path


def download_huggingface_dataset(repo_id: str, output_path: Path) -> Path:
    """
    Download a dataset from HuggingFace Hub.

    Args:
        repo_id: HuggingFace dataset repository ID (e.g., 'jswartz/nex-real-forward-facing')
        output_path: Directory to download the dataset to

    Returns:
        Path to the downloaded dataset
    """
    print(f"Downloading dataset from HuggingFace: {repo_id}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(output_path),
    )

    print(f"Dataset downloaded to: {output_path}")
    return output_path
