# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from omegaconf import DictConfig

all_presets = {
    "ks-sdf": DictConfig(
        {
            "url": "https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/checkpoints/ks.pth",
            "voxel_size": 0.1,
            "kernel_dim": 4,
            "tree_depth": 4,
            "adaptive_depth": 2,
            "unet": {"f_maps": 32},
            "udf": {"enabled": True},
            "interpolator": {"n_hidden": 2, "hidden_dim": 16},
            "density_range": [1.0, 20.0],
        }
    ),
}
