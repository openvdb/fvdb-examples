# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
import pathlib
import time
from typing import Literal

import torch
import tyro
from fvdb.viz import Scene

from garfvdb.config import (
    GaussianSplatSegmentationTrainingConfig,
    SfmSceneSegmentationTransformConfig,
)
from garfvdb.training.segmentation import GaussianSplatScaleConditionedSegmentation
from garfvdb.training.segmentation_writer import (
    GaussianSplatSegmentationWriter,
    GaussianSplatSegmentationWriterConfig,
)
from garfvdb.util import load_sfm_scene, load_splats_from_file


def main(
    sfm_dataset_path: pathlib.Path,
    reconstruction_path: pathlib.Path,
    config: GaussianSplatSegmentationTrainingConfig = GaussianSplatSegmentationTrainingConfig(),
    tx: SfmSceneSegmentationTransformConfig = SfmSceneSegmentationTransformConfig(),
    io: GaussianSplatSegmentationWriterConfig = GaussianSplatSegmentationWriterConfig(),
    dataset_type: Literal["colmap", "simple_directory", "e57"] = "colmap",
    run_name: str | None = None,
    log_path: pathlib.Path | None = pathlib.Path("garfvdb_logs"),
    use_every_n_as_val: int = -1,
    device: str | torch.device = "cuda",
    log_every: int = 10,
    visualize_every: int = -1,
    verbose: bool = False,
):
    """
    Train a Gaussian splat segmentation model from a set of images and camera poses.

    Args:
        sfm_dataset_path (pathlib.Path): Path to the dataset. For "colmap" datasets, this should be the
            directory containing the `images` and `sparse` subdirectories. For "simple_directory" datasets,
            this should be the directory containing the images and a `cameras.txt` file. For "e57" datasets,
            this should be the path to the E57 file or directory.
        reconstruction_path (pathlib.Path): Path to the precomputed Gaussian splat reconstruction to load.
        config (GaussianSplatSegmentationTrainingConfig): Configuration for the Gaussian splat segmentation training.
        tx (SfmSceneSegmentationTransformConfig): Configuration for the transforms to apply to the SfmScene before training.
        io (GaussianSplatSegmentationWriterConfig): Configuration for saving segmentation metrics and checkpoints.
        dataset_type (Literal["colmap", "simple_directory", "e57"]): Type of dataset to load.
        run_name (str | None): Name of the run. If None, a name will be generated based on the current date and time.
        log_path (pathlib.Path | None): Path to log metrics and checkpoints. If None, no metrics or checkpoints will be saved.
        use_every_n_as_val (int): Use every n-th image as a validation image. If -1, do not use a separate validation split.
        device (str | torch.device): Device to use for training.
        log_every (int): Log training metrics every n steps.
        visualize_every (int): Update the viewer every n epochs. If -1, do not visualize.
        verbose (bool): Whether to log debug messages.
    """
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s : %(message)s")
    logger = logging.getLogger(__name__)

    # Load the SfmScene
    sfm_scene = load_sfm_scene(sfm_dataset_path, dataset_type)

    # Load the GaussianSplat3D model
    gs_model, metadata = load_splats_from_file(reconstruction_path, device)
    normalization_transform = metadata.get("normalization_transform", None)

    if visualize_every > 0:
        viewer = Scene(name="Gaussian Splat Segmentation Visualization")
    else:
        viewer = None

    writer = GaussianSplatSegmentationWriter(run_name=run_name, save_path=log_path, config=io, exist_ok=False)
    runner = GaussianSplatScaleConditionedSegmentation.new(
        sfm_scene=tx.build_scene_transforms(gs_model, normalization_transform)(sfm_scene),
        gs_model=gs_model,
        gs_model_path=reconstruction_path,
        writer=writer,
        config=config,
        device=device,
        use_every_n_as_val=use_every_n_as_val,
        viewer_update_interval_epochs=visualize_every,
        log_interval_steps=log_every,
    )

    runner.train()

    if viewer is not None:
        logger.info("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    tyro.cli(main)
