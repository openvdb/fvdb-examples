# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Writer for LangSplatV2 training output logging.

Handles CSV metrics, disk image saving, checkpoints, and optional TensorBoard
logging, following the GARfVDB ``GaussianSplatSegmentationWriter`` pattern.
"""
import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Any, TextIO

import cv2
import numpy as np
import torch


@dataclass
class LangSplatV2WriterConfig:
    """Configuration for what data gets saved during LangSplatV2 training.

    Controls which output channels are active (images, checkpoints, metrics,
    TensorBoard) and how they are buffered.
    """

    save_images: bool = False
    """Whether to save rendered images to disk."""

    save_checkpoints: bool = True
    """Whether to save model checkpoints to disk."""

    save_metrics: bool = True
    """Whether to save metrics to CSV file."""

    use_tensorboard: bool = False
    """Whether to log scalars (and optionally images) to TensorBoard."""

    save_images_to_tensorboard: bool = False
    """Whether to also log images to TensorBoard (requires ``use_tensorboard=True``)."""

    metrics_file_buffer_size: int = 8 * 1024 * 1024  # 8 MB
    """Write-buffer size for the metrics CSV file."""


class LangSplatV2Writer:
    """Logging and I/O handler for LangSplatV2 training.

    Manages directories, metric files, image saving, checkpoints, and
    optional TensorBoard integration.  Follows the same interface as
    GARfVDB's ``GaussianSplatSegmentationWriter``.

    Directory layout when ``save_path`` is provided::

        save_path/run_name/
            checkpoints/
                <step>/
                    langsplatv2_ckpt.pt
            images/
                <step>/
                    predicted_features.jpg
                    gt_features.jpg
                    ...
            tensorboard/
                events.out.tfevents...
            metrics_log.csv
    """

    def __init__(
        self,
        run_name: str | None,
        save_path: pathlib.Path | None,
        exist_ok: bool = False,
        config: LangSplatV2WriterConfig = LangSplatV2WriterConfig(),
    ) -> None:
        """Create a new writer instance.

        Args:
            run_name: Name of this training run.  If *None* and ``save_path``
                is provided, a timestamped name is generated automatically.
            save_path: Root directory for results.  If *None*, nothing is
                written to disk.
            exist_ok: If *True*, reuse an existing directory instead of
                raising ``FileExistsError``.
            config: Controls what gets saved.
        """
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._config = config

        # ---- resolve save path and run name ----
        if run_name is None and save_path is not None:
            save_path = save_path.resolve()
            run_name, save_path = self._make_unique_results_directory_based_on_time(save_path, prefix="run")
        elif run_name is None and save_path is None:
            save_path = None
            run_name = None
        elif run_name is not None and save_path is not None:
            save_path = (save_path / run_name).resolve()
            if not exist_ok and save_path.exists():
                raise FileExistsError(
                    f"Directory {save_path} already exists. Use exist_ok=True to overwrite."
                )
            save_path.mkdir(parents=True, exist_ok=exist_ok)
        else:
            # run_name given but no save_path -- use as tag only
            save_path = None

        self._run_name = run_name
        self._save_path = save_path

        # ---- checkpoints directory ----
        self._checkpoints_path: pathlib.Path | None = None
        if self._config.save_checkpoints and self._save_path is not None:
            self._checkpoints_path = self._save_path / "checkpoints"
            self._checkpoints_path.mkdir(parents=True, exist_ok=True)

        # ---- images directory ----
        self._images_path: pathlib.Path | None = None
        if self._config.save_images and self._save_path is not None:
            self._images_path = self._save_path / "images"
            self._images_path.mkdir(parents=True, exist_ok=True)

        # ---- metrics CSV ----
        self._metrics_log_file_handle: TextIO | None = None
        if self._config.save_metrics and self._save_path is not None:
            self._metrics_path = self._save_path / "metrics_log.csv"
            self._metrics_log_file_handle = open(
                self._metrics_path, "a", buffering=self._config.metrics_file_buffer_size
            )

        # ---- TensorBoard ----
        self._tb_writer = None
        if self._config.use_tensorboard and self._save_path is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_path = self._save_path / "tensorboard"
                tb_path.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(tb_path))
            except ImportError:
                self._logger.warning(
                    "TensorBoard logging is enabled but torch.utils.tensorboard is not available. "
                    "Disabling TensorBoard logging."
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_name(self) -> str | None:
        """Name of this training run, or *None*."""
        return self._run_name

    @property
    def log_path(self) -> pathlib.Path | None:
        """Directory where results are saved, or *None*."""
        return self._save_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close open file handles.  Safe to call multiple times."""
        if self._metrics_log_file_handle is not None and not self._metrics_log_file_handle.closed:
            self._metrics_log_file_handle.flush()
            self._metrics_log_file_handle.close()
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

    def flush(self) -> None:
        """Flush buffered data without closing handles."""
        if self._metrics_log_file_handle is not None and not self._metrics_log_file_handle.closed:
            self._metrics_log_file_handle.flush()
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "LangSplatV2Writer":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @torch.no_grad()
    def log_metric(
        self,
        global_step: int,
        metric_name: str,
        metric_value: float | int | torch.Tensor,
    ) -> None:
        """Log a scalar metric to CSV and/or TensorBoard.

        Args:
            global_step: Training step.
            metric_name: Metric tag (e.g. ``"train/loss"``).
            metric_value: Scalar value.
        """
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.detach().cpu().item()
        metric_value = float(metric_value)

        if self._config.save_metrics and self._metrics_log_file_handle is not None:
            self._metrics_log_file_handle.write(f"{global_step},{metric_name},{metric_value}\n")

        if self._config.use_tensorboard and self._tb_writer is not None:
            self._tb_writer.add_scalar(metric_name, metric_value, global_step)

    @torch.no_grad()
    def save_image(
        self,
        global_step: int,
        image_name: str,
        image: torch.Tensor,
        jpeg_quality: int = 98,
    ) -> None:
        """Save an image to disk and/or TensorBoard.

        Args:
            global_step: Training step.
            image_name: File name (may include subdirectory, e.g.
                ``"eval/beauty_render.jpg"``).  Must end in ``.png``,
                ``.jpg``, or ``.jpeg``.
            image: Image tensor of shape ``(H, W)``, ``(H, W, C)``, or
                ``(B, H, W, C)``.  Floating-point images are scaled to
                ``[0, 255]``.
            jpeg_quality: JPEG quality when saving ``.jpg`` files.
        """
        if not self._config.save_images and not (
            self._config.use_tensorboard and self._config.save_images_to_tensorboard
        ):
            return

        image = self._to_batched_uint8_image(image)  # (B, H, W, C)

        # ---- save to disk ----
        if self._config.save_images and self._images_path is not None:
            image_path = self._resolve_saved_file(
                base_path=self._images_path,
                global_step=global_step,
                file_name=image_name,
                file_type="Image",
                allowed_suffixes=(".png", ".jpg", ".jpeg"),
                default_suffix=".png",
            )

            batch_size: int = image.shape[0]
            num_channels: int = image.shape[-1]
            for b in range(batch_size):
                batch_path = (
                    image_path.parent / f"{image_path.stem}_{b:04d}{image_path.suffix}"
                    if batch_size > 1
                    else image_path
                )
                image_np = image[b].cpu().numpy()
                if num_channels == 1:
                    image_np = image_np[:, :, 0]

                if batch_path.suffix.lower() == ".png":
                    cv2.imwrite(str(batch_path), image_np)
                elif batch_path.suffix.lower() in (".jpg", ".jpeg"):
                    cv2.imwrite(
                        str(batch_path),
                        image_np,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )

        # ---- save to TensorBoard ----
        if (
            self._config.use_tensorboard
            and self._config.save_images_to_tensorboard
            and self._tb_writer is not None
        ):
            self._tb_writer.add_images(
                image_name,
                image.permute(0, 3, 1, 2).contiguous(),
                global_step,
            )

    @torch.no_grad()
    def save_checkpoint(
        self,
        global_step: int,
        checkpoint_name: str,
        checkpoint: dict[str, Any],
    ) -> None:
        """Save a training checkpoint to disk.

        Args:
            global_step: Training step.
            checkpoint_name: File name (must end in ``.pt`` or ``.pth``).
            checkpoint: Checkpoint dictionary.
        """
        if self._config.save_checkpoints and self._checkpoints_path is not None:
            ckpt_path = self._resolve_saved_file(
                base_path=self._checkpoints_path,
                global_step=global_step,
                file_name=checkpoint_name,
                file_type="Checkpoint",
                allowed_suffixes=(".pth", ".pt"),
                default_suffix=".pt",
            )
            torch.save(checkpoint, ckpt_path)
            self._logger.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Helpers (matching GARfVDB writer helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_batched_uint8_image(image: torch.Tensor) -> torch.Tensor:
        """Convert an image tensor to batched uint8 ``(B, H, W, C)``.

        Accepts ``(H, W)``, ``(H, W, C)``, or ``(B, H, W, C)`` inputs with
        floating-point or uint8 dtype.  Floating-point values are scaled to
        ``[0, 255]``.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a torch.Tensor")
        if image.ndim not in (2, 3, 4):
            raise ValueError("Image must be 2D (H, W), 3D (H, W, C) or 4D (B, H, W, C)")

        num_channels: int = image.shape[-1] if image.ndim in (3, 4) else 1
        if num_channels not in (1, 3, 4):
            raise ValueError(
                f"Image must have C=1, C=3, or C=4 channels. "
                f"Got shape {image.shape} (C={num_channels})."
            )

        if image.ndim == 2:
            image = image.unsqueeze(-1)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.is_floating_point():
            image = (image * 255.0).clip(0, 255).to(torch.uint8)

        if image.dtype != torch.uint8:
            raise ValueError(f"Image must be torch.uint8 or floating point, got {image.dtype}.")

        return image

    @staticmethod
    def _resolve_saved_file(
        base_path: pathlib.Path,
        global_step: int,
        file_name: str,
        file_type: str,
        allowed_suffixes: tuple[str, ...],
        default_suffix: str,
    ) -> pathlib.Path:
        """Resolve and validate the path for a file to be saved at a given step."""
        step_file_path = (base_path / f"{global_step:08d}" / file_name).resolve()

        if not step_file_path.is_relative_to(base_path):
            raise ValueError(f"{file_type} name {file_name} resolves outside {base_path.name}/.")

        if not step_file_path.parent.exists():
            step_file_path.parent.mkdir(parents=True, exist_ok=True)

        if step_file_path.suffix == "":
            step_file_path = step_file_path.with_suffix(default_suffix)

        if step_file_path.suffix not in allowed_suffixes:
            raise ValueError(
                f"{file_type} must have one of {allowed_suffixes}, got {step_file_path.suffix}."
            )

        return step_file_path

    def _make_unique_results_directory_based_on_time(
        self,
        base_path: pathlib.Path,
        prefix: str,
    ) -> tuple[str, pathlib.Path]:
        """Create a uniquely-named results directory under *base_path*."""
        max_attempts = 50
        run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        for attempt in range(max_attempts):
            log_path = base_path / run_name
            try:
                log_path.mkdir(exist_ok=False, parents=True)
                self._logger.info(f"Created log directory: {run_name}")
                return run_name, log_path
            except FileExistsError:
                run_name = f"{prefix}_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{attempt + 2:02d}"
                continue
        raise FileExistsError(
            f"Failed to create unique directory after {max_attempts} attempts."
        )
