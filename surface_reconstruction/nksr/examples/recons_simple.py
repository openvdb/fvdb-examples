# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from pathlib import Path

import point_cloud_utils as pcu
import requests
import torch

import nksr_fvdb

if __name__ == "__main__":
    device = torch.device("cuda:0")

    bunny_mesh_path = Path.home() / "data" / "bunny.ply"
    if not bunny_mesh_path.exists():
        bunny_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        res = requests.get("https://huggingface.co/heiwang1997/nksr-checkpoints/resolve/main/data/buda.ply")
        with bunny_mesh_path.open("wb") as f:
            f.write(res.content)

    bunny_v, bunny_n = pcu.load_mesh_vn(bunny_mesh_path)

    input_xyz = torch.from_numpy(bunny_v).float().to(device)
    input_normal = torch.from_numpy(bunny_n).float().to(device)

    reconstructor = nksr_fvdb.Reconstructor(device, preset="ks-sdf")
    field = reconstructor.reconstruct(input_xyz, input_normal, voxel_size=1.0)
    mesh = field.extract_primal_mesh(depth=1)

    # Do something with the mesh...
