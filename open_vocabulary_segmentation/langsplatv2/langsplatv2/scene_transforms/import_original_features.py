# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Import pre-computed features from the original LangSplatV2 repository.

The original LangSplatV2 ``preprocess.py`` saves per-image features as two
numpy files:

- ``{image_name}_f.npy``: CLIP feature vectors ``[N_total, 512]`` (float16)
- ``{image_name}_s.npy``: segmentation maps ``[4, H, W]`` (int32)

This transform reads those files and writes them into our SfmCache format,
acting as a drop-in replacement for the ``ComputeMultiScaleSAM2Masks`` +
``ComputeCLIPFeatures`` pipeline to test our pipeline's outputs.
"""
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from fvdb_reality_capture.sfm_scene import SfmCache, SfmScene
from fvdb_reality_capture.transforms import BaseTransform, transform


def _compute_lengths_from_seg_maps(
    seg_maps: np.ndarray,
    total_features: int,
) -> list[int]:
    """Derive per-level feature counts from the original's segmentation maps.

    The original concatenates features in order (default, s, m, l) and assigns
    globally-offset indices in each seg_map channel.  It guarantees that the
    **max** index in each non-empty channel equals
    ``cumulative_length[level] - 1`` (asserted in ``preprocess.py``).  We
    exploit this by recovering cumulative lengths from the max index per
    channel: ``cum[j] = max(channel_j) + 1``.

    This is robust to masks whose indices never appear in the seg_map (e.g.
    a mask at the start of a level with zero pixels assigned) because only
    the max -- which the original guarantees correct -- is used.

    Args:
        seg_maps: Array of shape ``[4, H, W]`` with int32 indices (-1 = none).
        total_features: Total number of features in the corresponding
            ``_f.npy`` file (used as a cross-check).

    Returns:
        List of 4 integers giving the number of features at each scale level.
    """
    cum: list[int] = []
    for level in range(4):
        channel = seg_maps[level]
        valid = channel[channel >= 0]
        if len(valid) == 0:
            cum.append(cum[-1] if cum else 0)
        else:
            cum.append(int(valid.max()) + 1)

    lengths = [cum[0]] + [cum[j] - cum[j - 1] for j in range(1, 4)]
    return lengths


@transform
class ImportOriginalLangSplatV2Features(BaseTransform):
    """Import pre-computed language features from the original LangSplatV2.

    Reads ``_f.npy`` / ``_s.npy`` file pairs produced by the original
    ``preprocess.py`` and writes them into the SfmCache in the same dict
    format that ``ComputeCLIPFeatures`` produces:

    .. code-block:: python

        {"features": Tensor, "seg_maps": Tensor, "lengths": Tensor}

    This allows the training pipeline to consume original features without
    any changes to the dataset or trainer code.
    """

    version = "1.0.2"

    def __init__(
        self,
        original_features_dir: Path | str,
        clip_n_dims: int = 512,
    ):
        """
        Args:
            original_features_dir: Path to the original ``language_features/``
                directory containing ``*_f.npy`` and ``*_s.npy`` files.
            clip_n_dims: Expected CLIP embedding dimensionality (used for
                cache naming and validation).
        """
        self._features_dir = Path(original_features_dir)
        self._clip_n_dims = clip_n_dims
        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    def __call__(self, input_scene: SfmScene) -> SfmScene:
        if len(input_scene.images) == 0:
            self._logger.warning("No images in SfmScene. Returning unchanged.")
            return input_scene

        input_cache: SfmCache = input_scene.cache

        version_safe = self.version.replace(".", "_")
        cache_prefix = (
            f"imported_original_langsplatv2_features"
            f"_{self._clip_n_dims}_v{version_safe}"
        )
        output_cache = input_cache.make_folder(
            cache_prefix,
            description="Imported features from original LangSplatV2",
        )

        num_zeropad = len(str(input_scene.num_images)) + 2
        regenerate_cache = False

        if output_cache.num_files != input_scene.num_images:
            if output_cache.num_files != 0:
                self._logger.info(
                    f"Cache has {output_cache.num_files} files but expected "
                    f"{input_scene.num_images}. Regenerating cache."
                )
            output_cache.clear_current_folder()
            regenerate_cache = True

        if not regenerate_cache:
            for image_id in range(input_scene.num_images):
                cache_filename = f"features_{image_id:0{num_zeropad}}"
                if not output_cache.has_file(cache_filename):
                    self._logger.info(
                        f"{cache_filename} missing from cache. Regenerating."
                    )
                    output_cache.clear_current_folder()
                    regenerate_cache = True
                    break

        if regenerate_cache:
            self._logger.info(
                f"Importing original features from {self._features_dir}"
            )
            pbar = tqdm.tqdm(
                input_scene.images,
                unit="imgs",
                desc="Importing original features",
            )

            for image_meta in pbar:
                image_name = Path(image_meta.image_path).stem
                f_path = self._features_dir / f"{image_name}_f.npy"
                s_path = self._features_dir / f"{image_name}_s.npy"

                if not f_path.exists():
                    raise FileNotFoundError(
                        f"Feature file not found: {f_path}. "
                        f"Run the original LangSplatV2 preprocess.py first."
                    )
                if not s_path.exists():
                    raise FileNotFoundError(
                        f"Segmentation map not found: {s_path}. "
                        f"Run the original LangSplatV2 preprocess.py first."
                    )

                features_np = np.load(str(f_path))  # [N_total, 512], float16
                seg_maps_np = np.load(str(s_path))  # [4, H, W], int32

                if features_np.shape[1] != self._clip_n_dims:
                    raise ValueError(
                        f"Feature dimension mismatch for {image_name}: "
                        f"expected {self._clip_n_dims}, got {features_np.shape[1]}"
                    )

                lengths = _compute_lengths_from_seg_maps(
                    seg_maps_np, features_np.shape[0]
                )

                expected_total = sum(lengths)
                if features_np.shape[0] != expected_total:
                    self._logger.warning(
                        f"{image_name}: feature count ({features_np.shape[0]}) "
                        f"!= sum(lengths) ({expected_total}). "
                        f"Possible corrupt feature/seg_map pair."
                    )

                features = torch.from_numpy(features_np).half()
                seg_maps = torch.from_numpy(seg_maps_np).int()
                lengths_t = torch.tensor(lengths, dtype=torch.int32)

                cache_filename = f"features_{image_meta.image_id:0{num_zeropad}}"
                output_cache.write_file(
                    name=cache_filename,
                    data={
                        "features": features,
                        "seg_maps": seg_maps,
                        "lengths": lengths_t,
                    },
                    data_type="pt",
                    metadata={
                        "source": "original_langsplatv2",
                        "clip_n_dims": self._clip_n_dims,
                        "original_features_dir": str(self._features_dir),
                    },
                )

            pbar.close()
            self._logger.info(
                f"Imported features for {input_scene.num_images} images."
            )
        else:
            self._logger.info("Original features already cached.")

        output_scene = SfmScene(
            cameras=input_scene.cameras,
            images=input_scene.images,
            points=input_scene.points,
            points_err=input_scene.points_err,
            points_rgb=input_scene.points_rgb,
            scene_bbox=input_scene.scene_bbox,
            transformation_matrix=input_scene.transformation_matrix,
            cache=output_cache,
        )

        return output_scene

    @staticmethod
    def name() -> str:
        return "ImportOriginalLangSplatV2Features"

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name(),
            "version": self.version,
            "original_features_dir": str(self._features_dir),
            "clip_n_dims": self._clip_n_dims,
        }

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, Any],
    ) -> "ImportOriginalLangSplatV2Features":
        if state_dict["name"] != "ImportOriginalLangSplatV2Features":
            raise ValueError(
                f"Expected 'ImportOriginalLangSplatV2Features', "
                f"got {state_dict['name']}"
            )
        return ImportOriginalLangSplatV2Features(
            original_features_dir=state_dict["original_features_dir"],
            clip_n_dims=state_dict["clip_n_dims"],
        )
