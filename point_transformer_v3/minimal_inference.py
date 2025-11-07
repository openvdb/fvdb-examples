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
            order_type=("z", "z-trans", "hilbert", "hilbert-trans"),
            # no_conv_in_cpe=True,
            # embedding_mode="linear",
        ).to(device)
    return model


def prepare_batched_inputs_from_scannet_points(batch_samples, voxel_size=0.1, device="cuda"):
    """Prepare batched inputs from a list of ScanNet-like samples.

    Args:
        batch_samples: list of dicts with keys "grid_coords" and "color".
        voxel_size: float
        device: torch.device or str

    Returns:
        grid: fvdb.GridBatch
        jfeats: fvdb.JaggedTensor with concatenated [ijk, color]
    """
    coords_list = [torch.tensor(s["grid_coords"], device=device, dtype=torch.int32) for s in batch_samples]
    colors_list = [torch.tensor(s["color"], device=device, dtype=torch.float32) for s in batch_samples]

    coords_jagged = fvdb.JaggedTensor(coords_list)
    grid = fvdb.GridBatch.from_ijk(
        coords_jagged,
        voxel_sizes=[[voxel_size, voxel_size, voxel_size]] * len(coords_list),
        origins=[0.0] * 3,
    )
    color_jagged = fvdb.JaggedTensor(colors_list)
    color_vdb_order = grid.inject_from_ijk(coords_jagged, color_jagged)
    jfeats = color_vdb_order.jdata
    # jfeats = fvdb.jcat([grid.ijk.float(), jfeats], dim=1)
    jfeats = fvdb.jcat([grid.ijk.float(), color_vdb_order], dim=1)
    return grid, jfeats
    # import pdb; pdb.set_trace()
    # jfeats = torch.cat([grid.ijk.float(), jfeats], dim=1)
    # return grid, fvdb.JaggedTensor(jfeats)


def main():

    parser = argparse.ArgumentParser(description="Minimal inference script for PT-v3 on ScanNet point cloud data")
    parser.add_argument(
        "--data-path", type=str, default="scannet_samples.json", help="Path to the scannet samples json file"
    )
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size for grid sampling")
    parser.add_argument("--patch-size", type=int, default=1024, help="Maximum points per sample")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of samples per forward pass")
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
    batch_size = int(args.batch_size)

    # Accumulate gradients across all batches
    accumulated_grad_last16 = None
    first_module_name_global = None
    for batch_start in range(0, len(scannet_data), batch_size):
        batch = scannet_data[batch_start : batch_start + batch_size]
        logger.info(f"--- Processing Batch {batch_start//batch_size + 1} with {len(batch)} samples ---")

        # Run inference
        logger.info("Running batched inference...")
        nvtx.range_push("inference")
        nvtx.range_push("create_batched_grid_from_points")
        init_grid, init_feat = prepare_batched_inputs_from_scannet_points(
            batch, voxel_size=args.voxel_size, device=device
        )
        nvtx.range_pop()
        grid, feats = model(init_grid, init_feat)
        nvtx.range_pop()

        # Compute per-sample forward stats by splitting with offsets
        offsets = feats.joffsets.to(device=feats.jdata.device, dtype=torch.int64)
        num_samples_in_batch = offsets.numel() - 1

        # Store per-sample forward statistics
        for local_idx in range(num_samples_in_batch):
            start = int(offsets[local_idx].item())
            end = int(offsets[local_idx + 1].item())
            j_slice = feats.jdata[start:end]
            sample_dict = batch[local_idx]

            # Per-sample forward stats (independent of batch size)
            statistics_to_save.append(
                {
                    "num_points": int(sample_dict.get("num_points", end - start)),
                    "output_feats_shape": [int(end - start), int(j_slice.shape[1]) if j_slice.ndim == 2 else 0],
                    "output_feats_sum": float(j_slice.sum().item()) if j_slice.numel() > 0 else 0.0,
                    "output_feats_last_element": float(j_slice[-1, -1].item()) if (end - start) > 0 else 0.0,
                    "loss": float(j_slice.sum().item()) if j_slice.numel() > 0 else 0.0,  # Per-sample loss
                }
            )

            logger.info(
                f"Sample {local_idx + 1}/{num_samples_in_batch}: feats.shape={j_slice.shape}, feats.sum()={j_slice.sum().item():.6f}, feats[last-element]={j_slice[-1, -1].item():.6f}"
            )

        # Test backward path - compute accumulated gradients for the entire batch
        logger.info("Testing backward path...")
        nvtx.range_push("backward")
        batch_loss = feats.jdata.sum()  # Total loss for the entire batch
        batch_loss.backward()
        nvtx.range_pop()

        # Collect gradient from the first module and accumulate across batches
        batch_first_module_grad_last16 = None

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Get the last 16 gradients from the first module with gradients
                if batch_first_module_grad_last16 is None:
                    if first_module_name_global is None:
                        first_module_name_global = name

                    if name == first_module_name_global:
                        grad_flat = param.grad.flatten()
                        if len(grad_flat) >= 16:
                            batch_first_module_grad_last16 = grad_flat[-16:].tolist()
                            break

        # Accumulate gradients across all batches
        if batch_first_module_grad_last16 is not None:
            if accumulated_grad_last16 is None:
                accumulated_grad_last16 = batch_first_module_grad_last16[:]
            else:
                accumulated_grad_last16 = [
                    a + b for a, b in zip(accumulated_grad_last16, batch_first_module_grad_last16)
                ]

        logger.info(f"Batch loss: {batch_loss.item():.6f}")
        if batch_first_module_grad_last16:
            logger.info(
                f"Batch gradients from {first_module_name_global}: {[f'{x:.6f}' for x in batch_first_module_grad_last16[:4]]}...{[f'{x:.6f}' for x in batch_first_module_grad_last16[-4:]]}"
            )
        if accumulated_grad_last16:
            logger.info(
                f"Accumulated gradients: {[f'{x:.6f}' for x in accumulated_grad_last16[:4]]}...{[f'{x:.6f}' for x in accumulated_grad_last16[-4:]]}"
            )

        model.zero_grad()

    # Create final output structure with global gradient info separate from per-sample stats
    output_data = {
        "global_stats": {
            "first_module_grad_last16": accumulated_grad_last16,
            "first_module_name": first_module_name_global,
            "total_samples": len(statistics_to_save),
            "batch_size": batch_size,
        },
        "per_sample_stats": statistics_to_save,
    }

    # save the statistics to a json file
    output_file = args.data_path.replace(".json", f"_output.json")
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    logger.info(f"Statistics saved to {output_file}")

    # Log final accumulated gradient summary
    if accumulated_grad_last16:
        grad_sum = sum(accumulated_grad_last16)
        grad_norm = sum(x * x for x in accumulated_grad_last16) ** 0.5
        logger.info(
            f"Final accumulated gradient from {first_module_name_global}: sum={grad_sum:.6f}, norm={grad_norm:.6f}"
        )


if __name__ == "__main__":
    main()

## Example commands:
# scannet_samples_small.json
# python minimal_inference.py --data-path data/scannet_samples_small.json --voxel-size 0.1 --patch-size 1024 --batch-size 1

# scannet_samples_large.json
# python minimal_inference.py --data-path data/scannet_samples_large.json --voxel-size 0.02 --patch-size 1024 --batch-size 1
