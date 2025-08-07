# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Minimal inference script for PT-v3 on ScanNet point cloud data.
This script demonstrates how to:
1. Load point cloud data from scannet_samples.json
2. Load and run the PT-v3 model
"""

import argparse
import gc
import json
import logging
import os

import numpy as np
import torch
from model import PTV3

import fvdb

try:
    import torch.cuda.nvtx as nvtx

    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

    class DummyNVTX:
        def range_push(self, msg):
            pass

        def range_pop(self):
            pass

    nvtx = DummyNVTX()


def create_ptv3_model(args, device, num_classes):
    """Create a PT-v3 model.

    Args:
        args: Arguments object containing model configuration.
        device: Device to place the model on.
        num_classes: Number of semantic classes.

    Returns:
        A PTV3 model instance.
    """
    if args.model_mode == "enc":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 2),
            enc_channels=(32, 64, 128, 256),
            enc_num_heads=(1, 1, 1, 1),
            patch_size=args.patch_size,
            proj_drop=0.1,
        ).to(device)
    elif args.model_mode == "encdec":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 2),
            enc_channels=(32, 64, 128, 256),
            enc_num_heads=(1, 1, 1, 1),
            dec_depths=(2, 2, 2),
            dec_channels=(128, 64, 32),
            dec_num_heads=(1, 1, 1),
            patch_size=args.patch_size,
            proj_drop=0.1,
        ).to(device)
    elif args.model_mode == "enc_multihead":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 2),
            enc_channels=(32, 64, 128, 256),
            enc_num_heads=(1, 2, 4, 8),
            patch_size=args.patch_size,
            proj_drop=0.1,
        ).to(device)
    elif args.model_mode == "encdec_multihead":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 2),
            enc_channels=(32, 64, 128, 256),
            enc_num_heads=(1, 2, 4, 8),
            dec_depths=(2, 2, 2),
            dec_channels=(128, 64, 32),
            dec_num_heads=(4, 2, 1),
            patch_size=args.patch_size,
            proj_drop=0.1,
        ).to(device)
    elif args.model_mode == "encdec_multihead_large" or args.model_mode == "encdec_multihead_large_new_kmap":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_heads=(2, 4, 8, 16, 32),
            # enc_num_heads=(1,1,1,1,1), # Raise an error: "RuntimeError: FlashAttention forward only supports head dimension at most 256"
            dec_depths=(2, 2, 2, 2),
            dec_channels=(256, 128, 64, 64),
            dec_num_heads=(16, 8, 4, 4),
            # dec_num_heads=(1,1,1,1),
            patch_size=args.patch_size,
            proj_drop=0.1,
        ).to(device)
    elif args.model_mode == "encdec_multihead_large_droppath":
        model = PTV3(
            num_classes=num_classes,
            input_dim=6,  # xyz + rgb color
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_heads=(2, 4, 8, 16, 32),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(256, 128, 64, 64),
            dec_num_heads=(16, 8, 4, 4),
            patch_size=args.patch_size,
            proj_drop=0.0,
            drop_path=0.3,
        ).to(device)
    return model


def prepare_input_from_scannet_points(color, grid_coords, voxel_size=0.1, device="cuda"):
    """Prepare input from scannet points.

    Args:
        color: Color of the points.
        grid_coords: Grid coordinates of the points.
        voxel_size: Voxel size for grid sampling.
        device: Device to place the tensors on.

    Returns:
        grid: GridBatch of the given point cloud.
        jfeats: JaggedTensor of the point cloud features.
    """
    # Convert to torch tensors
    grid_coords_tensor = torch.tensor(grid_coords, device=device, dtype=torch.int32)
    color_tensor = torch.tensor(color, device=device, dtype=torch.float32)

    # Create jagged tensor for grid coordinates
    coords_jagged = fvdb.JaggedTensor([grid_coords_tensor])

    # Create grid from coordinates
    grid = fvdb.GridBatch.from_ijk(coords_jagged, voxel_sizes=[[voxel_size, voxel_size, voxel_size]], origins=[0.0] * 3)
    color_jdata = fvdb.JaggedTensor([color_tensor])
    color_vdb_order = grid.inject_from_ijk(coords_jagged, color_jdata)

    # Create features tensor (coordinates + color)
    jfeats = color_vdb_order.jdata
    jfeats = fvdb.jcat([grid.ijk.float(), jfeats], dim=1)

    return grid, jfeats


def main():

    parser = argparse.ArgumentParser(description="Minimal inference script for PT-v3 on ScanNet point cloud data")
    parser.add_argument(
        "--data-path", type=str, default="scannet_samples.json", help="Path to the scannet samples json file"
    )
    parser.add_argument("--voxel-size", type=float, default=0.1, help="Voxel size for grid sampling")
    parser.add_argument("--patch-size", type=int, default=0, help="Maximum points per sample")
    parser.add_argument(
        "--model-mode", type=str, default="encdec_multihead_large_droppath", help="The model configuration to choose."
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.data_path.replace(".json", ".log")),
            logging.StreamHandler(),
        ],  # Also log to console
    )

    logger = logging.getLogger(__name__)

    gc.disable()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # Set random seed for reproducibility

    scannet_data = json.load(open(args.data_path, "r"))

    # Model parameters
    num_classes = 20  # Number of semantic classes
    # Initialize PT-v3 model
    logger.info("Initializing PT-v3 model...")
    model = create_ptv3_model(args, device, num_classes)

    # Initialize model with random weights. Print the name of the parameter. And the last few elements of the parameter flattened.
    with torch.no_grad():
        for name, param in model.named_parameters():
            # reset the seed to easily compare the weights difference.
            torch.manual_seed(42)
            np.random.seed(42)
            if len(param.shape) == 5:  # For the convolution weights.
                shape = (param.data.shape[0], 3, 3, 3, param.data.shape[1])
                tmp = torch.empty(shape, device=device)
                tmp.normal_(mean=0, std=0.01)
                param.data = tmp.permute(0, 4, 3, 2, 1).contiguous()
            else:
                param.data.normal_(mean=0, std=0.01)
            logger.info(f"Parameter {name}: {param.shape}. Last 3 elems: {param.flatten()[-3:]}")

    torch.manual_seed(42)
    np.random.seed(42)
    # Process each sample
    logger.info("Using fvdb-based ptv3 model.")
    statistics_to_save = []
    for sample_idx, sample in enumerate(scannet_data):
        logger.info(f"--- Processing Sample {sample_idx + 1}/{len(scannet_data)} ---")

        # Extract data from sample
        num_points = sample["num_points"]
        grid_coords = np.array(sample["grid_coords"])
        color = np.array(sample["color"])
        label = np.array(sample["label"]) if "label" in sample else None

        logger.info(f"Sample {sample_idx + 1}: {num_points} points")

        # Run inference
        logger.info("Running inference...")
        nvtx.range_push("inference")
        nvtx.range_push("create_grid_from_points")
        init_grid, init_feat = prepare_input_from_scannet_points(color, grid_coords, voxel_size=0.1, device=device)
        nvtx.range_pop()
        grid, feats = model(init_grid, init_feat)  # outputs is a dict with keys "grid" and "feats". It is not logits.
        nvtx.range_pop()

        # Test backward path
        logger.info("Testing backward path...")
        nvtx.range_push("backward")

        # Create a dummy loss (sum of output features)
        loss = feats.jdata.sum()

        # Backward pass
        loss.backward()
        nvtx.range_pop()

        # Collect gradient statistics
        grad_stats = {}
        total_grad_norm = 0.0
        num_params_with_grad = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[f"{name}_grad_norm"] = grad_norm
                grad_stats[f"{name}_grad_sum"] = param.grad.sum().item()
                grad_stats[f"{name}_grad_last_element"] = param.grad.flatten()[-1].item()
                total_grad_norm += grad_norm**2
                num_params_with_grad += 1
            else:
                grad_stats[f"{name}_grad_norm"] = 0.0
                grad_stats[f"{name}_grad_sum"] = 0.0
                grad_stats[f"{name}_grad_last_element"] = 0.0

        total_grad_norm = total_grad_norm**0.5  # L2 norm

        # Log the statistics of the output features and gradients
        logger.info(
            f"feats.shape: {feats.jdata.shape}. feats.sum(): {feats.jdata.sum().item()}. feats[last-element]: {feats.jdata[-1, -1].item()}"
        )
        logger.info(f"Loss: {loss.item()}")
        logger.info(f"Total gradient norm: {total_grad_norm}")
        logger.info(f"Parameters with gradients: {num_params_with_grad}")

        statistics_to_save.append(
            {
                "num_points": num_points,
                "output_feats_shape": feats.jdata.shape,
                "output_feats_sum": feats.jdata.sum().item(),
                "output_feats_last_element": feats.jdata[-1, -1].item(),
                "loss": loss.item(),
                "total_grad_norm": total_grad_norm,
                "num_params_with_grad": num_params_with_grad,
                "gradient_stats": grad_stats,
            }
        )

        # Clear gradients for next iteration
        model.zero_grad()

    # save the statistics to a json file
    output_file = args.data_path.replace(".json", f"_output.json")
    with open(output_file, "w") as f:
        json.dump(statistics_to_save, f, indent=4)
    logger.info(f"Statistics saved to {output_file}")


if __name__ == "__main__":
    main()

## Example commands:
# scannet_samples_small.json
# python minimal_inference.py --data-path data/scannet_samples_small.json --voxel-size 0.1 --patch-size 1024 --model-mode encdec_multihead_large_droppath

# scannet_samples_large.json
# python minimal_inference.py --data-path data/scannet_samples_large.json --voxel-size 0.02 --patch-size 1024 --model-mode encdec_multihead_large_droppath
