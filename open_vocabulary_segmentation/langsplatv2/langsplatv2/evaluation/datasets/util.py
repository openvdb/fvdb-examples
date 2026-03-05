# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
from zipfile import ZipFile


def download_google_drive_file(file_id: str, output_path: Path, filename: str = "download.zip") -> Path:
    """
    Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID (from the sharing URL).
        output_path: Directory to save and extract to.
        filename: Name for the downloaded file.

    Returns:
        Path to the output directory after extraction.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download files from Google Drive. "
            "Install it with: pip install gdown"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    zip_path = output_path / filename

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive (file_id={file_id})...")
    gdown.download(url, str(zip_path), quiet=False)

    if zip_path.suffix == ".zip":
        print("Extracting archive...")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(output_path)
        # Clean up zip file
        zip_path.unlink()
        print(f"Extracted to: {output_path}")

    return output_path
