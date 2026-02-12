# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Training runner for LangSplatV2 language feature learning.

Manages the complete training pipeline including dataset creation, codebook
initialization, optimization loop, checkpointing, and evaluation.
"""
import logging
import math
import os
import random
from typing import Any, Callable

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import tqdm
from fvdb import GaussianSplat3d
from fvdb_reality_capture.sfm_scene import SfmScene

from ..config import LangSplatV2ModelConfig, LangSplatV2TrainingConfig
from ..loss import calculate_langsplatv2_loss
from ..model import LangSplatV2Model
from ..util import calculate_pca_projection, cosine_error_map, pca_projection_fast
from ..vq_utils import (
    ResidualVectorQuantizationWithClustering,
    load_clip_features_for_level,
)
from .dataset import (
    InfiniteSampler,
    LangSplatV2CollateFn,
    LangSplatV2Dataset,
    build_feature_map,
)
from .langsplatv2_writer import LangSplatV2Writer, LangSplatV2WriterConfig

logger = logging.getLogger(__name__)


class LangSplatV2Trainer:
    """Trainer for LangSplatV2 language feature learning.

    Manages the complete lifecycle: dataset preparation, codebook initialization
    via K-means, training loop with rendering and loss computation,
    checkpointing, and evaluation.

    This class follows the same pattern as GARfVDB's
    ``GaussianSplatScaleConditionedSegmentation`` for consistency.
    """

    version = "0.1.0"

    __PRIVATE__ = object()

    def __init__(
        self,
        model: LangSplatV2Model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        config: LangSplatV2TrainingConfig,
        gs_model: GaussianSplat3d,
        gs_model_path: "os.PathLike[str]",
        sfm_scene: SfmScene,
        train_dataset: LangSplatV2Dataset,
        val_dataset: LangSplatV2Dataset | None,
        writer: LangSplatV2Writer,
        start_step: int,
        log_interval_steps: int,
        viz_callback: Callable[["LangSplatV2Trainer", int], None] | None = None,
        _private: object | None = None,
    ) -> None:
        """Initialize the training runner.

        Note:
            This constructor should only be called via ``new()`` or
            ``from_state_dict()``.
        """
        if _private is not LangSplatV2Trainer.__PRIVATE__:
            raise ValueError("LangSplatV2Trainer should only be initialized through `new` or `from_state_dict`.")

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

        self._cfg = config
        self._start_step = start_step
        self._sfm_scene = sfm_scene
        self._gs_model = gs_model
        self._gs_model_path = gs_model_path

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset

        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler

        self._global_step: int = start_step

        self._writer = writer
        self._log_interval_steps = log_interval_steps
        self._viz_callback = viz_callback

    def close(self) -> None:
        """Flush and close the writer.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        self._writer.close()

    def __del__(self) -> None:
        # Only flush -- the writer may be shared, so let the caller close it.
        self._writer.flush()

    def __enter__(self) -> "LangSplatV2Trainer":
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object) -> None:
        self.close()

    @property
    def model(self) -> LangSplatV2Model:
        """Get the LangSplatV2 model."""
        return self._model

    @property
    def gs_model(self) -> GaussianSplat3d:
        """Get the underlying GaussianSplat3d model."""
        return self._gs_model

    @property
    def config(self) -> LangSplatV2TrainingConfig:
        """Get the training configuration."""
        return self._cfg

    @property
    def device(self) -> torch.device:
        """Get the device the model is on."""
        return self._model.device

    @property
    def total_steps(self) -> int:
        """Get the total number of steps for training."""
        steps_per_epoch = math.ceil(len(self._train_dataset) / self._cfg.batch_size)
        computed_total_steps: int = self._cfg.max_epochs * steps_per_epoch
        total_steps: int = self._cfg.max_steps if self._cfg.max_steps is not None else computed_total_steps
        return total_steps

    @classmethod
    def new(
        cls,
        sfm_scene: SfmScene,
        gs_model: GaussianSplat3d,
        gs_model_path: "os.PathLike[str]",
        writer: LangSplatV2Writer,
        config: LangSplatV2TrainingConfig = LangSplatV2TrainingConfig(),
        device: str | torch.device = "cuda",
        use_every_n_as_val: int = -1,
        log_interval_steps: int = 10,
        viz_callback: Callable[["LangSplatV2Trainer", int], None] | None = None,
        cache_dataset: bool = True,
    ) -> "LangSplatV2Trainer":
        """Create a new LangSplatV2 training run.

        This method handles the full initialization:
        1. Splits dataset into train/val
        2. Loads CLIP features for the configured scale level
        3. Initializes codebooks via K-means clustering
        4. Creates the model and optimizer

        Args:
            sfm_scene: Preprocessed SfmScene with CLIP features in cache.
            gs_model: Pre-trained GaussianSplat3d model.
            gs_model_path: Path to the Gaussian splat checkpoint.
            writer: Writer instance for logging metrics, images, and
                checkpoints.
            config: Training configuration.
            device: Device for training.
            use_every_n_as_val: Use every N-th image for validation.
                -1 = no validation split.
            log_interval_steps: How often to log metrics.
            viz_callback: Optional visualization callback.
            cache_dataset: Whether to cache data in memory.

        Returns:
            Initialized LangSplatV2Training instance.
        """
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Split into train/val
        indices = np.arange(sfm_scene.num_images)
        if use_every_n_as_val > 0:
            mask = np.ones(len(indices), dtype=bool)
            mask[::use_every_n_as_val] = False
            train_indices = indices[mask]
            val_indices = indices[~mask]
        else:
            train_indices = indices
            val_indices = np.array([], dtype=int)

        # Create datasets
        train_dataset = LangSplatV2Dataset(
            sfm_scene=sfm_scene,
            feature_level=config.feature_level,
            clip_n_dims=config.model.clip_n_dims,
            dataset_indices=train_indices,
            cache_features=cache_dataset,
            cache_images=cache_dataset,
        )
        val_dataset = None
        if len(val_indices) > 0:
            val_dataset = LangSplatV2Dataset(
                sfm_scene=sfm_scene,
                feature_level=config.feature_level,
                clip_n_dims=config.model.clip_n_dims,
                dataset_indices=val_indices,
                cache_features=cache_dataset,
                cache_images=cache_dataset,
            )

        logger.info(
            f"Dataset: {len(train_dataset)} training images, "
            f"{len(val_indices)} validation images, "
            f"feature_level={config.feature_level}"
        )

        # Initialize codebooks via K-means on CLIP features
        logger.info("Loading CLIP features for codebook initialization...")
        all_dataset = LangSplatV2Dataset(
            sfm_scene=sfm_scene,
            feature_level=config.feature_level,
            clip_n_dims=config.model.clip_n_dims,
            cache_features=False,
            cache_images=False,
        )
        clip_features = load_clip_features_for_level(
            full_dataset=all_dataset,
            feature_level=config.feature_level,
        )
        logger.info(f"Loaded {clip_features.shape[0]:,} CLIP features of dimension {clip_features.shape[1]}")

        # Run K-means to get initial codebooks
        logger.info(
            f"Initializing codebooks via K-means: "
            f"{config.model.vq_layer_num} levels, "
            f"{config.model.codebook_size} clusters, "
            f"dim={config.model.clip_n_dims}"
        )
        rvq = ResidualVectorQuantizationWithClustering(
            num_levels=config.model.vq_layer_num,
            num_clusters=config.model.codebook_size,
            feature_dim=config.model.clip_n_dims,
            device=device,
            seed=config.seed,
        )
        rvq.fit_quantizers(clip_features)
        initial_codebooks = torch.stack(rvq.quantizers, dim=0).to(device)
        logger.info(f"Codebook initialization complete: {list(initial_codebooks.shape)}")

        # Create model
        model = LangSplatV2Model(
            gs_model=gs_model,
            vq_layer_num=config.model.vq_layer_num,
            codebook_size=config.model.codebook_size,
            clip_n_dims=config.model.clip_n_dims,
            topk=config.model.topk,
            device=device,
        )

        # Initialize codebooks from K-means
        model.initialize_codebooks(initial_codebooks)

        logger.info(f"Model initialized with {gs_model.num_gaussians:,} Gaussians")

        # Create optimizer (only optimize language feature parameters)
        optimizer = torch.optim.Adam(
            params=[model.logits, model.codebooks],
            lr=config.learning_rate,
            eps=1e-15,
        )

        # No scheduler needed for constant LR (matching original LangSplatV2)
        scheduler = None

        return cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            gs_model=gs_model,
            gs_model_path=gs_model_path,
            sfm_scene=sfm_scene,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            writer=writer,
            start_step=0,
            log_interval_steps=log_interval_steps,
            viz_callback=viz_callback,
            _private=cls.__PRIVATE__,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        gs_model: GaussianSplat3d,
        gs_model_path: "os.PathLike[str]",
        sfm_scene: SfmScene | None = None,
        writer: LangSplatV2Writer | None = None,
        device: str | torch.device = "cuda",
        eval_only: bool = False,
    ) -> "LangSplatV2Trainer":
        """Load a training runner from a checkpoint.

        Args:
            state_dict: Checkpoint dictionary from ``state_dict()``.
            gs_model: GaussianSplat3d model.
            gs_model_path: Path to the GS model file.
            sfm_scene: Optional SfmScene. If None, loaded from checkpoint.
            writer: Optional writer for logging.  If None, a default writer
                with no output is used.
            device: Device for the model.
            eval_only: If True, disable gradients for evaluation.

        Returns:
            Restored LangSplatV2Trainer instance.
        """
        raw_config = state_dict["config"]
        # Handle nested model config (stored as dict in checkpoint)
        model_cfg = raw_config.pop("model", {})
        if isinstance(model_cfg, dict):
            model_cfg = LangSplatV2ModelConfig(**model_cfg)
        raw_config["model"] = model_cfg
        config = LangSplatV2TrainingConfig(**raw_config)
        global_step = state_dict["step"]

        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Restore SfmScene
        if sfm_scene is None:
            sfm_scene = SfmScene.from_state_dict(state_dict["sfm_scene"])

        # Restore model
        model = LangSplatV2Model.from_state_dict_with_config(
            state_dict["model"], gs_model, device
        )

        if eval_only:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

        # Restore optimizer
        optimizer = torch.optim.Adam(
            params=[model.logits, model.codebooks],
            lr=config.learning_rate,
            eps=1e-15,
        )
        if not eval_only:
            optimizer.load_state_dict(state_dict["optimizer"])

        # Restore datasets
        train_indices = np.array(state_dict.get("train_indices", []), dtype=int)
        val_indices = np.array(state_dict.get("val_indices", []), dtype=int)

        train_dataset = LangSplatV2Dataset(
            sfm_scene=sfm_scene,
            feature_level=config.feature_level,
            clip_n_dims=config.model.clip_n_dims,
            dataset_indices=train_indices if len(train_indices) > 0 else None,
        )
        val_dataset = None
        if len(val_indices) > 0:
            val_dataset = LangSplatV2Dataset(
                sfm_scene=sfm_scene,
                feature_level=config.feature_level,
                clip_n_dims=config.model.clip_n_dims,
                dataset_indices=val_indices,
            )

        # Use a null writer if none provided
        if writer is None:
            writer = LangSplatV2Writer(run_name=None, save_path=None)

        logger.info(f"Restored from checkpoint at step {global_step}")

        return cls(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            config=config,
            gs_model=gs_model,
            gs_model_path=gs_model_path,
            sfm_scene=sfm_scene,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            writer=writer,
            start_step=global_step,
            log_interval_steps=10,
            _private=cls.__PRIVATE__,
        )

    def train(self, show_progress: bool = True) -> None:
        """Run the training loop.

        Trains the language feature model by rendering sparse coefficient
        weight maps and comparing the decoded CLIP features against ground
        truth features from the preprocessing pipeline.

        Args:
            show_progress: Whether to display a progress bar.
        """
        if self._optimizer is None:
            raise ValueError("This runner was not created with an optimizer.")

        self._logger.info(
            f"Starting training for {self._cfg.max_epochs} epochs "
            f"({self.total_steps} total steps)"
        )
        if self._cfg.max_steps is not None:
            self._logger.info(f"  max_steps override: {self._cfg.max_steps}")
        self._logger.info(
            f"  Feature level: {self._cfg.feature_level}, "
            f"Batch size: {self._cfg.batch_size}, "
            f"LR: {self._cfg.learning_rate}"
        )
        self._logger.info(
            f"  Loss: cosine={self._cfg.use_cosine_loss}, "
            f"l1={self._cfg.use_l1_loss}, "
            f"normalize={self._cfg.normalize_features}"
        )

        # Warmup cache before creating DataLoader workers
        with nvtx.range("dataset_warmup_cache"):
            self._train_dataset.warmup_cache()

        # Configure DataLoader
        dataset_size = len(self._train_dataset)
        if dataset_size <= self._cfg.batch_size * 2:
            num_workers = 2
        elif dataset_size <= 64:
            num_workers = min(4, os.cpu_count() or 4)
        else:
            num_workers = min(8, os.cpu_count() or 4)

        sampler = InfiniteSampler(self._train_dataset, shuffle=True, seed=self._cfg.seed)
        prefetch = 2 if num_workers > 0 else None
        if num_workers > 0 and dataset_size < self._cfg.batch_size * 4:
            prefetch = 1

        trainloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._cfg.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=True,
            prefetch_factor=prefetch,
            collate_fn=LangSplatV2CollateFn,
        )

        self._model.train()
        self._optimizer.zero_grad()

        pbar = tqdm.tqdm(
            total=self.total_steps,
            desc="Training LangSplatV2",
            initial=self._start_step,
            disable=not show_progress,
        )

        # Gradient accumulation
        accum_steps = self._cfg.accumulate_grad_steps
        accumulated_loss: float = 0.0

        # Determine current VQ layer index based on training progress
        # Following LangSplatV2: layer_idx = min(step / 10000 * layer_num, layer_num - 1)
        layer_num = self._cfg.model.vq_layer_num

        # Track epochs by optimizer steps, accounting for batch size
        steps_per_epoch = math.ceil(len(self._train_dataset) / self._cfg.batch_size)
        prev_epoch = -1

        trainloader_iter = iter(trainloader)
        step = 0
        while True:
            try:
                minibatch = next(trainloader_iter)
            except StopIteration:
                break

            # Move compact data to device (features ~1MB + seg_map ~8MB,
            # NOT the dense [H,W,512] feature map which would be ~4GB)
            with nvtx.range("data_to_device"):
                minibatch = minibatch.to(self.device)

            # Build dense feature map on GPU from compact data
            with nvtx.range("build_feature_map"):
                gt_features, feature_mask = build_feature_map(
                    features=minibatch["features"],
                    seg_map=minibatch["seg_map"],
                    clip_n_dims=self._cfg.model.clip_n_dims,
                )

            img_w = minibatch["image_w"][0]
            img_h = minibatch["image_h"][0]

            # Current VQ layer index
            layer_idx = min(
                int(self._global_step / 10000 * layer_num),
                layer_num - 1,
            )

            # Forward pass
            with nvtx.range("forward_pass"):
                predicted_features, alpha = self._model(
                    world_to_camera=minibatch["world_to_camera"],
                    projection=minibatch["projection"],
                    image_width=img_w,
                    image_height=img_h,
                    layer_idx=layer_idx,
                )

            # Compute loss
            with nvtx.range("loss_computation"):
                loss_dict = calculate_langsplatv2_loss(
                    predicted_features=predicted_features,
                    gt_features=gt_features,
                    mask=feature_mask,
                    use_cosine_loss=self._cfg.use_cosine_loss,
                    use_l1_loss=self._cfg.use_l1_loss,
                    normalize_features=self._cfg.normalize_features,
                )
                loss = loss_dict["total_loss"]

            # Scale loss by accumulation steps
            loss = loss / accum_steps
            accumulated_loss += loss.item()

            # Zero gradients at the beginning of accumulation cycle
            if step % accum_steps == 0:
                self._optimizer.zero_grad()

            # Backward pass
            with nvtx.range("backward_pass"):
                loss.backward()

            # Optimizer step after accumulating gradients
            if (step + 1) % accum_steps == 0:
                with nvtx.range("optimizer_step"):
                    self._optimizer.step()
                    if self._scheduler is not None:
                        self._scheduler.step()
                accumulated_loss = 0.0

            self._global_step += 1

            # Logging
            display_loss = loss.item() * accum_steps
            pbar.set_postfix(
                loss=f"{display_loss:.4g}",
                level=self._cfg.feature_level,
                layer=layer_idx,
            )
            if self._global_step % self._log_interval_steps == 0:
                self._writer.log_metric(self._global_step, "train/loss", display_loss)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
                    self._writer.log_metric(self._global_step, "train/mem_gb", mem_gb)

                for key, val in loss_dict.items():
                    if key != "total_loss":
                        self._writer.log_metric(self._global_step, f"train/{key}", val.item())

                # Log training images when enabled
                if self._cfg.log_test_images:
                    self._log_training_images(
                        predicted_features.detach(),
                        gt_features.detach(),
                        feature_mask,
                    )

            pbar.update(1)

            # Calculate current epoch from steps processed
            epoch = self._global_step // steps_per_epoch

            # Check for epoch boundary - run save/eval only once per epoch transition
            if epoch > prev_epoch:
                prev_epoch = epoch

                # Visualization callback at epoch boundaries
                if self._viz_callback is not None:
                    try:
                        self._viz_callback(self, epoch)
                    except Exception as e:
                        self._logger.warning(f"Visualization callback failed: {e}")

                # Save checkpoint at configured epoch percentages
                if epoch in [pct * self._cfg.max_epochs // 100 for pct in self._cfg.save_at_percent]:
                    self._logger.info(f"Saving checkpoint at epoch {epoch} (step {self._global_step})")
                    self._writer.save_checkpoint(
                        self._global_step, "langsplatv2_ckpt.pt", self.state_dict()
                    )

                # Run evaluation at configured epoch percentages
                if epoch in [pct * self._cfg.max_epochs // 100 for pct in self._cfg.eval_at_percent]:
                    if self._val_dataset is not None and len(self._val_dataset) > 0:
                        self._logger.info(f"Running evaluation at epoch {epoch} (step {self._global_step})")
                        self.eval()

            # Check if we've reached max_steps or max_epochs
            if self._cfg.max_steps is not None and self._global_step >= self._cfg.max_steps:
                self._logger.info(f"Reached max steps: {self._cfg.max_steps}")
                break
            if epoch >= self._cfg.max_epochs:
                self._logger.info(f"Reached max epochs: {self._cfg.max_epochs}")
                break

            step += 1

        pbar.close()
        self._writer.flush()
        self._logger.info("Training completed.")

    @torch.no_grad()
    def _log_training_images(
        self,
        predicted_features: torch.Tensor,
        gt_features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> None:
        """Log PCA projections, feature coverage, and error heatmap during training.

        Args:
            predicted_features: Predicted CLIP features ``[B, H, W, C]``.
            gt_features: Ground truth CLIP features ``[B, H, W, C]``.
            feature_mask: Boolean validity mask ``[B, H, W]``.
        """
        try:
            # Compute shared PCA basis from GT features
            V = calculate_pca_projection(gt_features, n_components=3)

            # PCA projections (first sample in batch)
            pred_pca = pca_projection_fast(predicted_features, V=V, mask=feature_mask)
            gt_pca = pca_projection_fast(gt_features, V=V, mask=feature_mask)

            # Feature coverage mask as grayscale RGB
            coverage = feature_mask[0].float().unsqueeze(-1).expand(-1, -1, 3)  # [H, W, 3]

            # Cosine error heatmap
            error = cosine_error_map(predicted_features, gt_features, mask=feature_mask)

            self._writer.save_image(self._global_step, "train/predicted_features.jpg", pred_pca[0].cpu())
            self._writer.save_image(self._global_step, "train/gt_features.jpg", gt_pca[0].cpu())
            self._writer.save_image(self._global_step, "train/feature_coverage.jpg", coverage.cpu())
            self._writer.save_image(self._global_step, "train/cosine_error.jpg", error[0].cpu())
        except Exception as e:
            self._logger.warning(f"Failed to log training images: {e}")

    @torch.inference_mode()
    def eval(self) -> float:
        """Run evaluation on the validation set.

        Computes the average loss across all validation views, then renders
        a full set of diagnostic images for the first validation view:
        beauty render, PCA feature projections, cosine error heatmap,
        alpha map, and a side-by-side comparison composite.

        Returns:
            Average loss on the validation set.
        """
        if self._val_dataset is None or len(self._val_dataset) == 0:
            self._logger.warning("No validation dataset available.")
            return 0.0

        self._model.eval()

        valloader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=LangSplatV2CollateFn,
        )

        total_loss = 0.0
        num_batches = 0

        for val_batch in valloader:
            val_batch = val_batch.to(self.device)

            gt_features, feature_mask = build_feature_map(
                features=val_batch["features"],
                seg_map=val_batch["seg_map"],
                clip_n_dims=self._cfg.model.clip_n_dims,
            )

            img_w = val_batch["image_w"][0]
            img_h = val_batch["image_h"][0]

            predicted_features, alpha = self._model(
                world_to_camera=val_batch["world_to_camera"],
                projection=val_batch["projection"],
                image_width=img_w,
                image_height=img_h,
            )

            loss_dict = calculate_langsplatv2_loss(
                predicted_features=predicted_features,
                gt_features=gt_features,
                mask=feature_mask,
                use_cosine_loss=self._cfg.use_cosine_loss,
                use_l1_loss=self._cfg.use_l1_loss,
                normalize_features=self._cfg.normalize_features,
            )

            total_loss += loss_dict["total_loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self._logger.info(f"Evaluation loss: {avg_loss:.4f}")
        self._writer.log_metric(self._global_step, "eval/loss", avg_loss)

        # ---- Evaluation images for the first validation view ----
        try:
            self._log_evaluation_images(valloader)
        except Exception as e:
            self._logger.warning(f"Failed to log evaluation images: {e}")

        self._writer.flush()
        self._model.train()
        return avg_loss

    @torch.inference_mode()
    def _log_evaluation_images(self, valloader: torch.utils.data.DataLoader) -> None:
        """Render and save a full set of evaluation images for the first validation view.

        Produces: beauty render, predicted feature PCA, GT feature PCA,
        cosine error heatmap, alpha map, and a side-by-side comparison.

        Args:
            valloader: DataLoader over the validation set.
        """
        val_batch = next(iter(valloader)).to(self.device)

        gt_features, feature_mask = build_feature_map(
            features=val_batch["features"],
            seg_map=val_batch["seg_map"],
            clip_n_dims=self._cfg.model.clip_n_dims,
        )

        img_w = val_batch["image_w"][0]
        img_h = val_batch["image_h"][0]

        # Beauty render (RGB reference image)
        beauty, _ = self._gs_model.render_images(
            world_to_camera_matrices=val_batch["world_to_camera"],
            projection_matrices=val_batch["projection"],
            image_width=img_w,
            image_height=img_h,
            near=0.01,
            far=1e10,
        )
        beauty = beauty.clamp(0.0, 1.0)  # [B, H, W, 3]

        # Forward pass for predicted features
        predicted_features, alpha = self._model(
            world_to_camera=val_batch["world_to_camera"],
            projection=val_batch["projection"],
            image_width=img_w,
            image_height=img_h,
        )

        # Compute shared PCA basis from GT features
        V = calculate_pca_projection(gt_features, n_components=3)

        # PCA projections
        pred_pca = pca_projection_fast(predicted_features, V=V, mask=feature_mask)
        gt_pca = pca_projection_fast(gt_features, V=V, mask=feature_mask)

        # Cosine error heatmap
        error = cosine_error_map(predicted_features, gt_features, mask=feature_mask)

        # Alpha map as grayscale RGB
        alpha_rgb = alpha.clamp(0.0, 1.0).expand(-1, -1, -1, 3)  # [B, H, W, 3]

        # Side-by-side composite: GT PCA | Predicted PCA | Error
        comparison = torch.cat([gt_pca, pred_pca, error], dim=2)  # [B, H, 3*W, 3]

        # Save all images (first sample in batch)
        self._writer.save_image(self._global_step, "eval/beauty_render.jpg", beauty[0].cpu())
        self._writer.save_image(self._global_step, "eval/predicted_features.jpg", pred_pca[0].cpu())
        self._writer.save_image(self._global_step, "eval/gt_features.jpg", gt_pca[0].cpu())
        self._writer.save_image(self._global_step, "eval/cosine_error.jpg", error[0].cpu())
        self._writer.save_image(self._global_step, "eval/alpha_map.jpg", alpha_rgb[0].cpu())
        self._writer.save_image(self._global_step, "eval/comparison.jpg", comparison[0].cpu())

    @torch.no_grad()
    def state_dict(self) -> dict[str, Any]:
        """Get the complete training state for checkpointing.

        Returns:
            Dictionary containing all state needed to resume training.
        """
        return {
            "magic": "LangSplatV2Checkpoint",
            "version": self.version,
            "step": self._global_step,
            "config": {
                "seed": self._cfg.seed,
                "feature_level": self._cfg.feature_level,
                "max_steps": self._cfg.max_steps,
                "max_epochs": self._cfg.max_epochs,
                "learning_rate": self._cfg.learning_rate,
                "batch_size": self._cfg.batch_size,
                "accumulate_grad_steps": self._cfg.accumulate_grad_steps,
                "use_cosine_loss": self._cfg.use_cosine_loss,
                "use_l1_loss": self._cfg.use_l1_loss,
                "normalize_features": self._cfg.normalize_features,
                "log_test_images": self._cfg.log_test_images,
                "eval_at_percent": self._cfg.eval_at_percent,
                "save_at_percent": self._cfg.save_at_percent,
                "model": {
                    "vq_layer_num": self._cfg.model.vq_layer_num,
                    "codebook_size": self._cfg.model.codebook_size,
                    "clip_n_dims": self._cfg.model.clip_n_dims,
                    "topk": self._cfg.model.topk,
                },
            },
            "sfm_scene": self._sfm_scene.state_dict(),
            "gs_model_path": str(self._gs_model_path),
            "model": self._model.state_dict_with_config(),
            "optimizer": self._optimizer.state_dict(),
            "train_indices": self._train_dataset._indices.tolist()
            if hasattr(self._train_dataset, "_indices")
            else [],
            "val_indices": self._val_dataset._indices.tolist()
            if self._val_dataset is not None and hasattr(self._val_dataset, "_indices")
            else [],
        }
