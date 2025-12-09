# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""
Fully connected ResNet v2-style block for point-wise feature transformation.

- Operates on feature vectors (no spatial structure).
- Uses pre-activation: ReLU → Linear → ReLU → Linear.
- No batch normalization (suited to irregular point clouds).
- Residual branch is zero-initialized, so the block starts as identity/projection.

Reference:
  He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016.
"""


import torch
from fvdb.types import DeviceIdentifier, resolve_device


class ResnetBlockFC(torch.nn.Module):
    """Fully connected ResNet block with bottleneck and zero-initialized residual.

    Args:
        size_in: Input feature dimension.
        size_out: Output feature dimension (defaults to size_in).
    """

    size_in: int
    size_hidden: int
    size_out: int
    fully_connected_0: torch.nn.Linear
    fully_connected_1: torch.nn.Linear
    shortcut: torch.nn.Linear | None

    def __init__(
        self,
        size_in: int,
        size_out: int | None = None,
        *,
        dtype: torch.dtype = torch.float32,
        device: DeviceIdentifier = "cpu",
    ) -> None:
        """Initialize the ResNet block.

        Args:
            size_in: Input feature dimension.
            size_out: Output feature dimension. If None, defaults to size_in
                (identity-shaped block).
            dtype: Data type to use for the module. Defaults to torch.float32.
            device: Device to place the module on. Defaults to "cpu".
        """
        t_device: torch.device = resolve_device(device)
        super().__init__()

        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_hidden = min(size_in, size_out)
        self.size_out = size_out

        # Submodules
        self.fully_connected_0 = torch.nn.Linear(self.size_in, self.size_hidden, dtype=dtype, device=t_device)
        self.fully_connected_1 = torch.nn.Linear(self.size_hidden, self.size_out, dtype=dtype, device=t_device)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Linear(size_in, size_out, bias=False, dtype=dtype, device=t_device)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize or reinitialize learnable parameters.

        The residual branch (fully_connected_1) is initialized to zero so that the block
        starts as identity-like. Other layers use PyTorch's default Kaiming
        uniform initialization.
        """
        # fully_connected_0 and shortcut use default PyTorch initialization (Kaiming uniform)
        self.fully_connected_0.reset_parameters()
        if self.shortcut is not None:
            self.shortcut.reset_parameters()

        # Zero-initialize fully_connected_1 so residual branch outputs zero at init
        # This makes the block behave like identity (or projection) initially
        torch.nn.init.zeros_(self.fully_connected_1.weight)
        torch.nn.init.zeros_(self.fully_connected_1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _hidden: torch.Tensor = self.fully_connected_0(torch.nn.functional.relu(x))
        residual: torch.Tensor = self.fully_connected_1(torch.nn.functional.relu(_hidden, inplace=True))
        return residual + (x if self.shortcut is None else self.shortcut(x))

    def extra_repr(self) -> str:
        return f"size_in={self.size_in}, size_hidden={self.size_hidden}, size_out={self.size_out}"
