# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from typing import Any, Callable, Literal, Union

import fvdb
import numpy as np
import torch
from fvdb import GaussianSplat3d
from sklearn.preprocessing import QuantileTransformer

from garfvdb.config import GARfVDBModelConfig
from garfvdb.training.dataset import GARfVDBInput
from garfvdb.util import rgb_to_sh


class SparseConvWithSkips(torch.nn.Module):
    """
    PyTorch module that implements an fVDB sparse convolutional network with skip connections.
    """

    def __init__(self, num_grids: int):
        super().__init__()

        self.kernel_size = 3
        self.relu = fvdb.nn.ReLU()

        # Initialize weights and ensure they require gradients
        for i in range(num_grids):
            for j in range(3):
                self.add_module(f"conv_{i}_{j}", fvdb.nn.SparseConv3d(8, 8, self.kernel_size, bias=False))
                torch.nn.init.xavier_uniform_(self.get_submodule(f"conv_{i}_{j}").weight)

    def forward(self, data: fvdb.JaggedTensor, grid_batch: fvdb.GridBatch) -> fvdb.JaggedTensor:
        result = []

        for i in range(len(data)):
            in_data = data[i]
            in_grid = grid_batch[i]

            plan = fvdb.ConvolutionPlan.from_grid_batch(kernel_size=self.kernel_size, stride=1, source_grid=in_grid)

            x = self.get_submodule(f"conv_{i}_0")(in_data, plan)

            out_grid = plan.target_grid

            x = self.relu(x, out_grid)

            x = self.get_submodule(f"conv_{i}_1")(x, plan)
            x = self.relu(x, out_grid)

            # # Third conv layer with skip connection
            x = self.get_submodule(f"conv_{i}_2")(x, plan)
            x.data.jdata += in_data.jdata

            result.append(x)

        return fvdb.JaggedTensor(result)


class GARfVDBModel(torch.nn.Module):
    """A PyTorch module that implements a segmentation method using Gaussian Splatting and ƒVDB.

    This model implements a segmentation method inspired by GARField which uses a trained Gaussian
    Splatting model as the renderable radiance field instead of a NeRF.  The encoded features are
    sampled from a batch of fVDB grids and then optionally passed through a sparse convolutional
    network before being concatenated with a scale value and passed to an MLP.  The MLP output is
    then used to predict a mask for each pixel.

    The model consists of several key components:
    1. A Gaussian Splatting model for 3D scene representation
    2. Feature encoding grids or per-gaussian features
    3. An optional sparse convolutional network for encoded feature processing
    4. An MLP for feature transformation

    Attributes:
        device (torch.device): The device to run the model on (CPU/GPU)
        model_config (GARfVDBModelConfig): Configuration parameters for the model
        gs_model (GaussianSplat3d): The underlying Gaussian Splatting model
        quantile_transformer (Callable): Function to normalize scales
        encoder_grids (Optional[fvdb.nn.VDBTensor]): Feature encoding grids when use_grid is True
        encoder_convnet (Optional[SparseConvWithSkips]): Sparse convolutional encoder network
        mlp (torch.nn.Sequential): MLP for feature transformation
    """

    mlp: torch.nn.Sequential
    encoder_gridbatch_features_data: torch.nn.Parameter | torch.Tensor
    encoder_gridbatch: fvdb.GridBatch
    encoder_convnet: SparseConvWithSkips | None

    def __init__(
        self,
        gs_model: GaussianSplat3d,
        scale_stats: torch.Tensor,
        model_config: GARfVDBModelConfig = GARfVDBModelConfig(),
        device: Union[str, torch.device] = torch.device("cuda"),
    ):
        """Initialize the GARfVDBModel from gsplat checkpoint and scale statistics from the entire training dataset.

        Args:
            gs_model: GaussianSplat3D model
            scale_stats: [N] Tensor of scale statistics from the entire training dataset
            model_config: Model configuration
            device: Device to use
        """

        super().__init__()
        self.register_state_dict_post_hook(self._state_dict_post_hook)

        self.device = device
        self.model_config = model_config
        self.gs_model = gs_model

        # Build the quantile transformer
        self.max_scale = torch.max(scale_stats)
        self.quantile_transformer = self._get_quantile_func(scale_stats)

        ###  Encoded Features ###
        # When `use_grid` is True, we will use the GARField method of encoding features at different scales using
        # a set of 3D feature grids at a range of scene scales.  At training time, these features are sampled from the
        # grids using the 3D centers of the rendered gaussians and weighted by their transmittance.
        # When `use_grid` is False, we will store the encoded features per-3d-guassian and render the features directly
        if self.model_config.use_grid:
            # GARField Encoder grids consist of two sets of 3D feature grids:
            # 1. 12 grids of number of voxels along each axis ranging from 16 -> 256
            # 2. 12 grids of number of voxels along each axis ranging from 256 -> 2048
            # Each grid has 8 feature channels
            resolution_range = [(16, 256), (256, 2048)]
            num_grids = [12, 12]
            # get the spatial extent of the means
            means = self.gs_model.means.detach()
            extent = means.max(dim=0).values - means.min(dim=0).values
            max_extent = extent.max().item()

            # Calculate the worldspace voxel sizes for each grid across the specified ranges
            voxel_sizes = []
            for res_range, n_grids in zip(resolution_range, num_grids):
                growth_rate = np.exp((np.log(res_range[1]) - np.log(res_range[0])) / (n_grids - 1))
                for i in range(n_grids):
                    res = np.round(res_range[0] * (growth_rate**i))
                    voxel_sizes.append([max_extent / res for _ in range(3)])

            # Create the encoder grids
            points = fvdb.JaggedTensor([means for _ in range(len(voxel_sizes))])
            self.encoder_gridbatch = fvdb.GridBatch.from_points(
                points=points,
                voxel_sizes=voxel_sizes,  # type: ignore
            )
            # The original GARField encoder grids store features at voxel corners
            self.encoder_gridbatch = self.encoder_gridbatch.dual_grid()

            # Initialize the encoded features
            enc_features = 1e-4 * torch.rand(
                [self.encoder_gridbatch.total_voxels, self.model_config.grid_feature_dim], device=device
            )
            # Store as a parameter to make it a leaf tensor
            self.encoder_gridbatch_features_data = torch.nn.Parameter(enc_features)

            if self.model_config.use_grid_conv:
                self.encoder_convnet = SparseConvWithSkips(num_grids=sum(num_grids)).to(device)

        else:
            # Initialize the per-gaussian features
            gs_features = torch.nn.Parameter(
                torch.zeros(
                    [self.gs_model.means.shape[0], 1, self.model_config.gs_features],
                    device=device,
                )
            )
            state_dict["sh0"] = gs_features
            self.gs_model.load_state_dict(state_dict)
            # have to set requires_grad here because 'load_state_dict' will set all params to the 'require_grad' value
            # from the state dict
            self.gs_model.sh0.requires_grad = True

        ### MLP ###
        # GARField MLP ('instance net') uses 4 hidden layers with 256 units each, 256 output channels
        # input channels is the concatenation of the encoder grids and a spatial scale encoding
        if self.model_config.use_grid:
            i_channels = self.model_config.grid_feature_dim * np.sum(num_grids) + 1
        else:
            i_channels = self.model_config.gs_features + 1
        o_channels = self.model_config.mlp_output_dim
        n_neurons = self.model_config.mlp_hidden_dim
        hidden_layers = self.model_config.mlp_num_layers

        # Create MLP
        mlp = torch.nn.Sequential(
            torch.nn.Linear(i_channels, n_neurons, bias=True),
            torch.nn.ReLU(),
            *[torch.nn.Linear(n_neurons, n_neurons, bias=True), torch.nn.ReLU()] * hidden_layers,
            torch.nn.Linear(n_neurons, o_channels, bias=False),
        )

        # Initialize the MLP weights before moving to device
        for layer in mlp:  # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        # Move to device and register
        self.mlp = mlp.to(device)

    @property
    def enc_features(self) -> fvdb.JaggedTensor:
        """Get the encoded features.

        Returns:
            fvdb.JaggedTensor: The encoded features
        """
        return self.encoder_gridbatch.jagged_like(self.encoder_gridbatch_features_data)

    def get_max_grouping_scale(self) -> torch.Tensor:
        """Get the maximum grouping scale.

        Returns:
            float: The maximum grouping world space scale
        """
        return self.max_scale

    @staticmethod
    def create_from_state_dict(
        state_dict: dict[str, Any],
        model_config: GARfVDBModelConfig,
        gs_model: GaussianSplat3d,
        scale_stats: torch.Tensor,
    ) -> "GARfVDBModel":
        """Load the model from the given state dictionary.

        Args:
            state_dict: State dictionary of the model
            model_config: Model configuration
            gs_model: GaussianSplat3d model
            scale_stats: Scale statistics

        Returns:
            GARfVDBModel: The loaded model
        """
        model = GARfVDBModel(gs_model, scale_stats, model_config, gs_model.device)

        if model_config.use_grid:
            model.encoder_gridbatch = state_dict["encoder_gridbatch"]
            model.encoder_gridbatch_features_data = torch.nn.Parameter(state_dict["encoder_gridbatch_features_data"])
        else:
            model.gs_model.sh0 = state_dict["sh0"]

        # Extract MLP state dict by filtering keys that start with "mlp." and stripping the prefix
        mlp_state_dict = {k.replace("mlp.", "", 1): v for k, v in state_dict.items() if k.startswith("mlp.")}
        model.mlp.load_state_dict(mlp_state_dict)
        return model

    # @staticmethod
    # def create_from_checkpoint(
    #     checkpoint_path: Union[str, Path],
    #     gs_model: GaussianSplat3d,
    #     scale_stats: torch.Tensor,
    #     device: Union[str, torch.device] = torch.device("cuda"),
    # ) -> "GARfVDBModel":
    #     """Load the model checkpoint from the given path.

    #     Args:
    #         checkpoint_path: Path to the checkpoint
    #     """
    #     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #     model_config = checkpoint["config"]["model"]
    #     model = GARfVDBModel(gs_model, scale_stats, model_config, device)

    #     if model_config.use_grid:
    #         model.encoder_gridbatch = checkpoint["encoder_gridbatch"]
    #         model.enc_features = checkpoint["encoder_features"]
    #     else:
    #         model.gs_model.sh0 = checkpoint["sh0"]
    #     model.mlp.load_state_dict(checkpoint["mlp"])
    #     return model

    def save_checkpoint(self, checkpoint_path: str):
        """Save the model checkpoint to the given path.

        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint = {}
        checkpoint["model_config"] = self.model_config
        if self.model_config.use_grid:
            checkpoint["encoder_gridbatch"] = self.encoder_gridbatch.cpu()
            checkpoint["encoder_features"] = self.enc_features.detach().cpu()
        else:
            checkpoint["sh0"] = self.gs_model.sh0.detach().cpu()
        checkpoint["mlp"] = self.mlp.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def get_quantile_transformer(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the scale quantile transformer.

        Returns:
            Callable: The quantile transformer
        """
        return self.quantile_transformer

    def _get_quantile_func(
        self, scales: torch.Tensor, distribution: Literal["uniform", "normal"] = "normal"
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Use 3D scale statistics to normalize scales with quantile transformer.

        Args:
            scales: [N] Tensor of scales
            distribution: Distribution to use for the quantile transformer

        Returns:
            Callable: The quantile transformer
        """
        scales = scales.flatten()
        scales = scales[(scales > 0) & (scales < self.get_max_grouping_scale().item())]

        scales = scales.detach().cpu().numpy()  # type: ignore

        # Calculate quantile transformer
        quantile_transformer = QuantileTransformer(output_distribution=distribution)
        quantile_transformer = quantile_transformer.fit(scales.reshape(-1, 1))

        def quantile_transformer_func(scales):
            # This function acts as a wrapper for QuantileTransformer.
            # QuantileTransformer expects a numpy array, while we have a torch tensor.
            # We need to preserve gradients, so we detach for the transform but then
            # re-attach gradients by using the original tensor in the computation
            scales_np = scales.detach().cpu().numpy()
            transformed_np = quantile_transformer.transform(scales_np)
            transformed = torch.from_numpy(transformed_np).to(scales.device, dtype=scales.dtype)

            # Create a differentiable path: use the transformed values but maintain gradient flow
            # by making it a function of the original scales. We do this by treating the
            # transform as approximately linear locally and using the identity gradient.
            # This is an approximation but allows training to proceed.
            if scales.requires_grad:
                # Detach the transformed values and re-attach to the computation graph
                # by adding a zero-gradient operation
                transformed = transformed + (scales - scales.detach())

            return transformed

        return quantile_transformer_func

    def get_encoded_features(self, input: GARfVDBInput) -> torch.Tensor:
        """Get the per-pixel encoder features for the given input images or pixel samples.

        Args:
            input: GARfVDBInput data

        Returns:
            Encoded feature tensor [B, R, S, F] or [B, H, W, S, F]
        """
        img_w = input["image_w"][0]
        img_h = input["image_h"][0]
        if not self.model_config.use_grid:
            intrinsics = input["projection"]
            world_to_cam = input["world_to_camera"]
            pixel_coords = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            features, opacity = self.gs_model.render_images(
                image_width=img_w,
                image_height=img_h,
                world_to_camera_matrices=world_to_cam,
                projection_matrices=intrinsics,
                near=0.01,
                far=1e10,
                sh_degree_to_use=0,
            )

            # # select just the cam_pts that we need from the whole image if pixel_coords is specified
            if pixel_coords is not None:
                batch_indices = torch.arange(features.shape[0]).view(-1, 1).expand(-1, pixel_coords.shape[1])
                features = features[batch_indices, pixel_coords[:, :, 0], pixel_coords[:, :, 1]]
                opacity = opacity[batch_indices, pixel_coords[:, :, 0], pixel_coords[:, :, 1]]
            return features
        else:
            intrinsics: torch.Tensor = input["projection"]
            world_to_cam: torch.Tensor = input["world_to_camera"]
            pixel_coords: torch.Tensor | None = input.get("pixel_coords", None)  # [B, rays_per_image, 2]

            # Render the IDs of the gaussians that contribute to each pixel, which we will use to sample the feature grids
            with torch.no_grad():
                if pixel_coords is not None:
                    ids, weights = self.gs_model.sparse_render_contributing_gaussian_ids(
                        pixels_to_render=fvdb.JaggedTensor(list(pixel_coords.unbind())),
                        top_k_contributors=self.model_config.depth_samples,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        image_width=img_w,
                        image_height=img_h,
                        near=0.01,
                        far=1e10,
                    )

                else:
                    ids, weights = self.gs_model.render_contributing_gaussian_ids(
                        top_k_contributors=self.model_config.depth_samples,
                        image_width=img_w,
                        image_height=img_h,
                        world_to_camera_matrices=world_to_cam,
                        projection_matrices=intrinsics,
                        near=0.01,
                        far=1e10,
                    )

                # TOOD:  There are Nans coming out of weights, need to fix this
                # if there's any nans print a warning
                if torch.isnan(weights.jdata).any():
                    logging.warning("WARNING: Nans in weights")
                    raise ValueError("Nans in contributing gaussian weights")
                    # weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

            if self.model_config.enc_feats_one_idx_per_ray:
                # Stochastically pick the depth_sample for each pixel based on the weights as probabilities
                # Convert weights to probabilities if they aren't already normalized

                # # Check for rays where all weights are 0
                # zero_mask = torch.tensor(weights.lshape) == 0  # [C,[R]]
                # if zero_mask.jdata.any():
                #     logging.warning(f"WARNING: Found {zero_mask.jdata.sum().item()} rays with all 0 weights")
                #     # For each ray with all zeros, set the first depth sample to 1e-10
                #     weights = torch.where(
                #         zero_mask.unsqueeze(-1),  # [B, R, 1] or [B, H, W, 1]
                #         torch.cat([torch.full_like(weights[..., :1], 1e-10), weights[..., 1:]], dim=-1),
                #         weights,
                #     )
                # Sample one index per ray based on probabilities
                # NOTE:  This probability would be much simpler if the division were just broadcastable
                probs = []
                # print("weights length: ", len(weights))
                # print("weights jsum length: ", len(weights.jsum(dim=0, keepdim=True)))
                for per_cam_weights, per_cam_weight_sum in zip(weights, weights.jsum(dim=0, keepdim=True)):
                    cam_probs = []

                    for per_ray_weights, per_ray_weight_sum in zip(per_cam_weights, per_cam_weight_sum):
                        if len(per_ray_weights) == 0:
                            cam_probs.append(torch.empty([0, per_ray_weights.eshape[-1]]))
                        else:
                            cam_probs.append(fvdb.relu(per_ray_weights.jdata / (per_ray_weight_sum.jdata + 1e-10)))
                    probs.append(cam_probs)
                probs = fvdb.JaggedTensor(probs)
                # shape: [B, R]
                depth_sample_indices = []
                for per_cam_probs in probs:
                    depth_sample_indices_cam = []
                    for per_ray_probs in per_cam_probs:
                        if len(per_ray_probs) == 0:
                            depth_sample_indices_cam.append(torch.empty([0]))
                        else:
                            depth_sample_indices_cam.append(torch.multinomial(per_ray_probs.jdata, num_samples=1))
                    depth_sample_indices.append(depth_sample_indices_cam)

                depth_sample_indices = fvdb.JaggedTensor(depth_sample_indices)
                # get 1 id per ray based on the depth_sample_indices
                single_ids = []
                for per_cam_ids, per_cam_depth_sample_indices in zip(ids, depth_sample_indices):

                    cam_ids = []
                    for per_ray_ids, per_ray_depth_sample_indices in zip(per_cam_ids, per_cam_depth_sample_indices):
                        if len(per_ray_ids) == 0:
                            cam_ids.append(torch.empty([0]))
                        else:
                            cam_ids.append(per_ray_ids.jdata[per_ray_depth_sample_indices.jdata])
                            # cam_ids.append(
                            #     torch.gather(per_ray_ids.jdata, dim=1, index=per_ray_depth_sample_indices.jdata)
                            # )
                    single_ids.append(cam_ids)
                ids = fvdb.JaggedTensor(single_ids)
                # ids = torch.gather(ids.squeeze(-1), dim=2, index=depth_sample_indices)  # [B, R, 1] or [B, H, W, 1]

            # unique ids
            ids_3dgs = ids % self.gs_model.means.shape[0]
            unique_ids, unique_ids_inverse = torch.unique(ids_3dgs.jdata, return_inverse=True)

            unique_world_pts = self.gs_model.means[unique_ids]

            # # debug test whether any of the points don't fall inside the grids' voxels
            # in_voxels = self.encoder_grids.grid.points_in_active_voxel(unique_world_pts)
            # num_not_in_voxels = (in_voxels.jdata == 0).sum().item()
            # if num_not_in_voxels > 0:
            #     logging.error(f"num_not_in_voxels: {num_not_in_voxels}")

            # sample the encoder grids at the unique world points
            enc_grid_sample_pts = fvdb.JaggedTensor(
                [unique_world_pts for _ in range(self.encoder_gridbatch.grid_count)]
            )
            if self.model_config.use_grid_conv:
                conv_output = self.encoder_convnet(self.enc_features, self.encoder_gridbatch)
                unique_enc_feats = conv_output.sample_trilinear(enc_grid_sample_pts)
            else:
                unique_enc_feats = self.encoder_gridbatch.sample_trilinear(enc_grid_sample_pts, self.enc_features)
            unique_cam_enc_feats = torch.cat(unique_enc_feats.unbind(), dim=-1)  # [unique_ids, F]
            # re-assemble the enc_feats based on the original ids
            enc_feats = unique_cam_enc_feats[unique_ids_inverse].squeeze(-2)  # [B, R, S, F]
            enc_feats = fvdb.JaggedTensor.from_data_offsets_and_list_ids(enc_feats, ids.joffsets, ids.jlidx)

            if not self.model_config.enc_feats_one_idx_per_ray:
                # Weighted sum of the enc_feats and transmittance weights
                # weights = weights.unsqueeze(-1)  # [B, R, S, 1] or [B, H, W, S, 1]

                # multiply by weights
                enc_feats.jdata = enc_feats.jdata * weights.jdata.unsqueeze(-1)
                enc_feats = enc_feats.jsum(dim=0, keepdim=True)
                # enc_feats = enc_feats.sum(dim=-2)  # [B, R, F] or [B, H, W, F]

            epsilon = 1e-6
            enc_feats.jdata = enc_feats.jdata / (torch.linalg.norm(enc_feats.jdata, dim=-1, keepdim=True) + epsilon)

            # Convert enc_feats to regular Tensor
            feature_size = enc_feats.eshape[-1]
            if pixel_coords is not None:
                # reshape enc_feats to [B, R, S, F]
                cam_batch_size, num_rays_per_cam = pixel_coords.shape[:2]
                return enc_feats.jdata.reshape(cam_batch_size, num_rays_per_cam, -1, feature_size)
            else:
                # reshape enc_feats to [B, H, W, S, F]
                return enc_feats.jdata.reshape(len(enc_feats), img_h, img_w, -1, feature_size)

    def get_mlp_output(self, enc_feats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Get the MLP output for the given encoder features and scales.

        Args:
            enc_feats: Encoder features [B, R, S, F] or [B, H, W, S, F]
            scales: Scales [B, R] or [B, H, W]

        Returns:
            MLP output [B, R, F] or [B, H, W, F]
        """
        epsilon = 1e-5

        # Debug print to understand shapes
        logging.debug(f"get_mlp_output shapes: enc_feats={enc_feats.shape}, scales={scales.shape}")

        # Process scales
        # Flatten for processing
        scales_flat = scales.reshape(-1)
        scales_flat = scales_flat.contiguous().view(-1, 1)
        scales_quant = self.quantile_transformer(scales_flat).to(scales_flat.device)
        scales_quant = scales_quant.reshape(enc_feats.shape[:-1]).unsqueeze(-1)

        in_feats = torch.cat([enc_feats, scales_quant], dim=-1)

        # Apply MLP
        gfeats = self.mlp(in_feats)
        norms = gfeats.norm(dim=-1, keepdim=True)
        gfeats = gfeats / (norms + epsilon)

        # Reshape back to 3D
        return gfeats

    def get_mask_output(self, input: GARfVDBInput, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the mask output for the given input and scale

        This is a convenience function used in inference or evaluation to get the mask output for a given scale.

        Args:
            input: GARfVDBInput data
            scale: Chosen scale for the masks

        Returns:
            Mask output [B, H, W, mlp_output_dim]
        """
        img_w = input["image_w"][0]
        img_h = input["image_h"][0]

        intrinsics = input["projection"]
        cam_to_world = input["camera_to_world"]
        world_to_cam = torch.linalg.inv(cam_to_world).contiguous()

        # Obtain per-gaussian features from the encoder grids
        world_pts = fvdb.JaggedTensor([self.gs_model.means for _ in range(self.encoder_gridbatch.grid_count)])
        if self.model_config.use_grid_conv:
            conv_output = self.encoder_convnet(self.encoder_grids)
            enc_feats = conv_output.sample_trilinear(world_pts)
        else:
            enc_feats = self.encoder_gridbatch.sample_trilinear(world_pts, self.enc_features)
        enc_feats = torch.cat(enc_feats.unbind(), dim=-1)  # [N, F]

        # Set them as the sh0 features
        enc_feats_state_dict = self.gs_model.state_dict()
        enc_feats_state_dict["sh0"] = rgb_to_sh(enc_feats).unsqueeze(1).contiguous()

        gs3d_enc_feats = GaussianSplat3d.from_state_dict(enc_feats_state_dict)

        # Render the image
        img_feats, alpha = gs3d_enc_feats.render_images(
            image_width=img_w,
            image_height=img_h,
            world_to_camera_matrices=world_to_cam,
            projection_matrices=intrinsics,
            near=0.01,
            far=1e10,
            sh_degree_to_use=0,
        )

        epsilon = 1e-6
        img_feats = img_feats / (torch.linalg.norm(img_feats, dim=-1, keepdim=True) + epsilon)

        # Apply MLP
        scales = torch.full_like(img_feats[..., :1], scale)
        gfeats = self.get_mlp_output(img_feats, scales)

        return gfeats, alpha

    @staticmethod
    def _state_dict_post_hook(module, state_dict, prefix, local_metadata):
        """Get the state dictionary of the model.

        Returns:
            dict: The state dictionary of the model.
        """
        state_dict["encoder_gridbatch"] = module.encoder_gridbatch.cpu()
