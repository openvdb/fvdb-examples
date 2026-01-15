from pathlib import Path
from typing import Optional

_dataset_root = None


def set_dataset_root(dataset_root: Path):
    global _dataset_root
    _dataset_root = dataset_root


def get_dataset_root() -> Optional[Path]:
    global _dataset_root
    if _dataset_root is None:
        return None
    _dataset_root.mkdir(parents=True, exist_ok=True)
    return _dataset_root


__all__ = ["set_dataset_root", "get_dataset_root"]
