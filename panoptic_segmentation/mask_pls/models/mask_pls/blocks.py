# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import fvdb
import fvdb.nn
from fvdb import ConvolutionPlan, GridBatch, JaggedTensor


class SelfAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        q_embed,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        q = k = self.with_pos_embed(q_embed, query_pos)
        q_embed2 = self.self_attn(q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask)[0]
        q_embed = q_embed + self.dropout(q_embed2)
        q_embed = self.norm(q_embed)
        return q_embed


class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def with_pos_embed2(self, tensor, pos: Optional[torch.Tensor]):
        out = torch.cat((tensor, pos.unsqueeze(0)), dim=-1)  # type: ignore
        return out

    def forward(
        self,
        q_embed,
        bb_feat,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
    ):
        q_embed = self.norm(q_embed)
        q_embed2 = self.multihead_attn(
            query=self.with_pos_embed(q_embed, query_pos),
            key=self.with_pos_embed(bb_feat, pos),
            value=self.with_pos_embed(bb_feat, pos),
            # value=bb_feat,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed


class FFNLayer(torch.nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm = torch.nn.LayerNorm(d_model)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim):
        super().__init__()
        self.num_layers = len(hidden_dim_list) + 1
        h = hidden_dim_list
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.relu(x)
        return x


class BasicConvolutionBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, bn_mom=0.1):
        super().__init__()
        if dilation != 1:
            raise NotImplementedError("Dilation not implemented for fVDB SparseConv3d")
        self.ks = ks
        self.stride = stride
        self.conv = fvdb.nn.SparseConv3d(inc, outc, kernel_size=ks, stride=stride)
        self.bn = fvdb.nn.BatchNorm(outc, momentum=bn_mom)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> Tuple[JaggedTensor, GridBatch]:
        target_grid = grid if self.stride == 1 else grid.conv_grid(kernel_size=self.ks, stride=self.stride)
        plan = ConvolutionPlan.from_grid_batch(
            kernel_size=self.ks, stride=self.stride, source_grid=grid, target_grid=target_grid
        )
        data = self.conv(data, plan)
        data = self.bn(data, target_grid)
        data = fvdb.relu(data)
        return data, target_grid


class BasicDeconvolutionBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, bn_mom=0.1):
        super().__init__()
        self.ks = ks
        self.stride = stride
        self.deconv = fvdb.nn.SparseConvTranspose3d(inc, outc, kernel_size=ks, stride=stride)
        self.bn = fvdb.nn.BatchNorm(outc, momentum=bn_mom)

    def forward(self, data: JaggedTensor, source_grid: GridBatch, target_grid: GridBatch) -> JaggedTensor:
        plan = ConvolutionPlan.from_grid_batch_transposed(
            kernel_size=self.ks, stride=self.stride, source_grid=source_grid, target_grid=target_grid
        )
        data = self.deconv(data, plan)
        data = self.bn(data, target_grid)
        data = target_grid.jagged_like(F.leaky_relu(data.jdata))
        return data


class ResidualBlock(torch.nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, bn_mom=0.1):
        super().__init__()
        if dilation != 1:
            raise NotImplementedError("Dilation not implemented for fVDB SparseConv3d")
        self.ks = ks
        self.stride = stride

        self.conv1 = fvdb.nn.SparseConv3d(inc, outc, kernel_size=ks, stride=stride)
        self.bn1 = fvdb.nn.BatchNorm(outc, momentum=bn_mom)
        self.conv2 = fvdb.nn.SparseConv3d(outc, outc, kernel_size=ks, stride=1)
        self.bn2 = fvdb.nn.BatchNorm(outc, momentum=bn_mom)

        if inc == outc and stride == 1:
            self.downsample_conv = None
            self.downsample_bn = None
        else:
            self.downsample_conv = fvdb.nn.SparseConv3d(inc, outc, kernel_size=1, stride=stride)
            self.downsample_bn = fvdb.nn.BatchNorm(outc, momentum=bn_mom)

    def forward(self, data: JaggedTensor, grid: GridBatch) -> Tuple[JaggedTensor, GridBatch]:
        target_grid = grid if self.stride == 1 else grid.conv_grid(kernel_size=self.ks, stride=self.stride)

        plan1 = ConvolutionPlan.from_grid_batch(
            kernel_size=self.ks, stride=self.stride, source_grid=grid, target_grid=target_grid
        )
        out = self.conv1(data, plan1)
        out = self.bn1(out, target_grid)
        out = fvdb.relu(out)

        plan2 = ConvolutionPlan.from_grid_batch(
            kernel_size=self.ks, stride=1, source_grid=target_grid, target_grid=target_grid
        )
        out = self.conv2(out, plan2)
        out = self.bn2(out, target_grid)

        if self.downsample_conv is not None:
            ds_plan = ConvolutionPlan.from_grid_batch(
                kernel_size=1, stride=self.stride, source_grid=grid, target_grid=target_grid
            )
            residual = self.downsample_conv(data, ds_plan)
            residual = self.downsample_bn(residual, target_grid)
        else:
            residual = data

        out = fvdb.relu(out + residual)
        return out, target_grid
