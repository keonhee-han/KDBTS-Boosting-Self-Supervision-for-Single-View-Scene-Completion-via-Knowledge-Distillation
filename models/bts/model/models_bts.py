"""
Main model implementation
"""

from typing import Dict, Optional
import torch
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn
import copy

from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.mlp_util import make_mlp

EPS = 1e-3

from models.common.model.multi_view_head import MultiViewHead
from models.common.backbones.ibrnet import IBRNetWithNeuRay

class BEVNet(torch.nn.Module):
    def __init__(
        self,
        conf,
        encoder: nn.Module,
        code_xyz,
        heads: Dict[str, nn.Module],
        final_pred_head: Optional[str] = None,
        ren_nc=None,
        code_token=None,
        **kwargs,
    ):  ## dependency injection
        super().__init__()
        ## model constructor's complexity
        self.encoder = encoder
        self.code_xyz = code_xyz
        self.code_token = code_token
        self.heads = nn.ModuleDict(heads)
        if final_pred_head:
            self.final_pred_head = final_pred_head
        else:
            self.final_pred_head = list(self.heads.keys())[0]  ### [('multiviewhead' | 'singleviewhead')][0] ## knowledge-distillation purpose
        self.img_height, self.img_width = kwargs.get("img_size")
        self.requires_bottleneck_feats = False

        for _, head in self.heads.items():
            if hasattr(head, "require_bottleneck_feats"):
                if head.require_bottleneck_feats and (
                    head.independent_token_net.__class__.__name__ == "NeuRayIndependentToken"
                ):  ## For read out token type: "NeuRayIndependentToken"
                    self.requires_bottleneck_feats = True
                    break
        self.use_viewdirs = conf.get("use_viewdirs", True)

        self.n_coarse = ren_nc
        # self.decoder_heads_conf = conf.get("decoder_heads")
        # self.loss_pgt = conf.get("loss_pgt")

        self.d_min, self.d_max = conf.get("z_near"), conf.get("z_far")
        self.learn_empty, self.empty_empty, self.inv_z = (
            conf.get("learn_empty", True),
            conf.get("empty_empty", False),
            conf.get("inv_z", True),
        )
        self.color_interpolation, self.code_mode = conf.get("color_interpolation", "bilinear"), conf.get(
            "code_mode", "z"
        )
        if self.code_mode not in ["z", "distance", "z_feat"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        ## For potential AE model (from GeoNeRF)
        ## config to compute total sample convoluted, ts_conv
        # self.ts_conv = (
        #     (ren_nc // 4) * (conf.get("patch_size") * conf.get("patch_size")) * (conf.get("ray_batch_size") // ren_nc)
        # )
        # self.AE = conf.get("AE")
        # ts_ = batch_size * (self.n_coarse // 4) * self.patch_size self.patch_size * (ray_batch_size // self.n_coarse)

        # move outside of the class
        # self.encoder = make_backbone(conf["encoder"])  ### ResNetEncoder + Monodepth2 as decoder
        # self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)
        self.flip_augmentation = conf.get("flip_augmentation", False)
        self.return_sample_depth = conf.get("return_sample_depth", False)
        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out  ### 64 + 39
        # d_in = 64
        d_out = 1 if self.sample_color else 4
        ## If sample_color is set to False, then d_out is set to 4 to represent the RGBA color values
        ## (red, green, blue, alpha) of the reconstructed scene. If sample_color is set to True, then d_out is set to 1
        ## to represent the estimated depth value of the reconstructed scene.

        self._d_in, self._d_out = d_in, d_out
        # self.mlp_coarse, self.mlp_fine = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out), make_mlp(
        #     conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True
        # )

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))
        ## factor to multiply the output of the corresponding MLP in the forward, which helps to control the range of the output values from the MLP
        self._scale = 0  ## set spatial resolution size accoridng to the scale of output feature map from the encoder
        self.invalid_features = None

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None, model_name=None):
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * 0.5 + 0.5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None
        ## Note: This is yet to be feature map before passing img to encoder
        n_, nv_, c_, h_, w_ = images_encoder.shape  ### [n_, nv_, 3:=RGB, 192, 640]
        c_l = self.encoder.latent_size  ### 64 c.f. paper D.1.

        if self.flip_augmentation and self.training:  ## data augmentation for color
            do_flip = (torch.rand(1) > 0.5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1,))

        if self.requires_bottleneck_feats:
            image_latents_ms, bottleneck_feats_ms = self.encoder(
                images_encoder.view(n_ * nv_, c_, h_, w_)
            )  ## Encoder of BTS's backbone model e.g. Monodepth2
        elif not self.requires_bottleneck_feats:
            image_latents_ms, _ = self.encoder(
                images_encoder.view(n_ * nv_, c_, h_, w_)
            )  ## Encoder of BTS's backbone model e.g. Monodepth2
            bottleneck_feats_ms = None
        else:
            NotImplementedError(f"__unrecognized input self.requires_bottleneck_feats:{self.requires_bottleneck_feats}")

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1,)) for il in image_latents_ms]
            if bottleneck_feats_ms is not None:
                bottleneck_feats_ms = [torch.flip(bf, dims=(-1,)) for bf in bottleneck_feats_ms]

        _, _, h_, w_ = image_latents_ms[
            0
        ].shape  ## get spatial resol from 1st layer out of 4 from feature maps generated by Enc
        image_latents_ms = [
            F.interpolate(image_latents, size=(h_, w_)).view(n_, nv_, c_l, h_, w_) for image_latents in image_latents_ms
        ]  ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer
        # img_feat_ms = [F.interpolate(feat_latents, size=(h_, w_)).view(n_, nv_, img_feat_ms[-1].shape[1], h_, w_) for feat_latents in img_feat_ms]    ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer
        if bottleneck_feats_ms is not None:
            self.grid_f_bottleneck_feats = F.interpolate(bottleneck_feats_ms[-1], size=(h_, w_)).view(
                n_, nv_, bottleneck_feats_ms[-1].shape[1], h_, w_
            )  ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer

        ## feature
        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        ## color
        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

        self.grid_t_pos = ids_encoder  ## positional embedding for token's order

    def sample_features(
        self,
        xyz,
        # BEV_feat_plane,
        use_single_featuremap=True
        # self, xyz, grid_t_pos, use_single_featuremap=True
    ):  ## 2nd arg: to control whether multiple feature maps should be combined into a single feature map or not. If True, the function will average the sampled features from multiple feature maps along the view dimension (nv) before returning the result. This can be useful when you want to combine information from multiple views or feature maps into a single representation.
        (
            n_,
            n_pts,
            _,
        ) = xyz.shape  ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        n_, nv_, c_, h_, w_ = self.grid_f_features[self._scale].shape  ### [B, nv, C, 192, 640]
        # if not use_single_featuremap:   nv_ = self.nv_
        xyz = xyz.unsqueeze(
            1
        )  # (n, 1, pts, 3)                            ## Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
        ones = torch.ones_like(
            xyz[..., :1]
        )  ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = (self.grid_f_poses_w2c[:, :nv_, :3, :]) @ xyz.permute(
            0, 1, 3, 2
        )  ## Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  ### [1, 2, 100000, 1]
        xyz_projected = (self.grid_f_Ks[:, :nv_] @ xyz_projected).permute(
            0, 1, 3, 2
        )  ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
        xy = xyz_projected[:, :, :, [0, 1]]  ## Extract the x,y coordinates and depth value from the projected points
        z_ = xyz_projected[:, :, :, 2:3]

        xy = xy / z_.clamp_min(
            EPS
        )  ## Normalize the x,y coordinates by the depth value and check for invalid points    => image coord -> pixel coord
        invalid = (
            (z_ <= EPS)
            | (xy[:, :, :, :1] < -1)
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )
        """given a vector p = (x, y, z) this is the difference of normalizing either:z ||p|| = sqrt(x^2 + y^2 + z^2). So you either give the network (x, y, z_normalized) or (x, y, ||p||_normalized) as input. It is just different parameterizations of the same point."""
        if (
            self.code_mode == "z"
        ):  ## Depending on the code mode, normalize the depth value or distance value to the [-1, 1] range and concatenate with the xy coordinates
            # Get z into [-1, 1] range  ## Normalizing the z coordinates leads to a consistent positional encoding of the viewing information. In line 172 the viewing information (xyz_projected) is given to a positional encoder before it is appended to the overall feature vector
            if self.inv_z:
                z_ = (1 / z_.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z_ = (z_ - self.d_min) / (self.d_max - self.d_min)
            z_ = 2 * z_ - 1
            xyz_projected = torch.cat((xy, z_), dim=-1)  ## concatenates the normalized x, y, and z coordinates

        elif self.code_mode == "z_feat":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1

            feat_map_pos_enc = torch.randn(distance.shape).to(distance.device)
            for i in range(distance.shape[1]):
                feat_map_pos_enc[:, i, :, :] = feat_map_pos_enc[:, i, :, :] + self.grid_t_pos[i]
            feat_map_pos_enc = nn.Parameter(feat_map_pos_enc)

            xyz_projected = torch.cat(
                (xy, distance, feat_map_pos_enc), dim=-1
            )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)

        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat(
                (xy, distance), dim=-1
            )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)
        xyz_code = (
            self.code_xyz(xyz_projected.view(n_ * nv_ * n_pts, -1)).view(n_, nv_, n_pts, -1).permute(0, 2, 1, 3)
        )  ## ! positional encoding dimension to check (concatenate)


        feature_map = self.grid_f_features[self._scale][:, :nv_]  ## !
        
        # Convert each sampled point from global coordinates to BEV grid coordinates
        # bev_grid_points = torch.stack([global_to_bev_grid(x, y) for x, y in ray_sample_points], dim=0)
        # bev_grid_points = bev_grid_points.unsqueeze(0).unsqueeze(2)  # Shape (1, n_samples, 1, 2)

        # These samples are from different scales
        if (
            self.learn_empty
        ):  ## "empty space" can refer to areas in a scene where there is no object, or it could also refer to areas that are not observed or are beyond the range of the sensor. This allows the model to have a distinct learned representation for "empty" space, which can be beneficial in tasks like 3D reconstruction where understanding both the objects in a scene and the empty space between them is important.
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c_).expand(
                n_, nv_, n_pts, c_
            )  ## trainable parameter, initialized with random features
        ## feature_map (2, 4, 64, 128, 128): n_ = 2, nv_ = 4 views, c_ = 64 channels in the feature map, and the height and width of the feature map are h = 128 and w = 128
        ## !TODO: for multiviews for F.grid_sample : xy.view(n_ * nv_, 1, -1, 2) To debug how xy looks like in order to integrate for multiview (by looking over doc in Pytorch regarding how to sample all frames)
        sampled_features = (
            F.grid_sample(
                feature_map.view(n_ * nv_, c_, h_, w_),
                xy.view(n_ * nv_, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .view(n_, nv_, c_, n_pts)
            .permute(0, 3, 1, 2)
        )  ## set x,y coordinates as grid feature

        if self.requires_bottleneck_feats:
            self.grid_f_bottleneck_feats = self.grid_f_bottleneck_feats[
                :, :nv_
            ]  ## taking last layer of encoder to feed into NeuRay
            sampled_bottleneck_feats = (
                F.grid_sample(
                    self.grid_f_bottleneck_feats.view(n_ * nv_, self.grid_f_bottleneck_feats.shape[2], h_, w_),
                    xy.view(n_ * nv_, 1, -1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .view(n_, nv_, self.grid_f_bottleneck_feats.shape[2], n_pts)
                .permute(0, 3, 1, 2)
            )
        else:
            sampled_bottleneck_feats = None  ### dim(sampled_features): (n_, nv_, n_pts, c_)

        if (
            self.learn_empty
        ):  ## Replace invalid features in the sampled features tensor with the corresponding features from the expanded empty feature
            sampled_features[invalid.expand(-1, -1, -1, c_)] = empty_feature_expanded[
                invalid.expand(-1, -1, -1, c_)
            ]  ## broadcasting and make it fit to feature map
        ## dim(xyz): (B,M), M:=#_pts.

        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)  ### (n_, nv_, M, C1+C_pos_emb)
        # sampled_features = torch.cat((sampled_features, xyz_code, token_pos_enc), dim=-1)  ### (n_, nv_, M, C1+C_pos_emb)

        return (
            sampled_features,
            invalid[..., 0].permute(0, 2, 1),
            sampled_bottleneck_feats,
        )  ## output BTS's decoder   ### img_feat_sampled.shape == (n_, n_pts, nv_, C_)

    def sample_colors(self, xyz):
        n_, n_pts, _ = xyz.shape  ## n := batch size, n_pts := #_points in world coord.
        n_, nv_, c_, h_, w_ = self.grid_c_imgs.shape  ## nv_ := #_views
        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(
            xyz[..., :1]
        )  ## create a tensor of ones with the same shape as the first two dimensions of (xyz), up to the third dimension, and add a trailing singleton dimension (shape: (n, 1)) e.g. (n, 1, pts, 1)
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  ## concatenates the tensor of ones with xyz along the last dimension (i.e., the dimension representing the coordinates of each point).
        xyz_projected = (self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(
            0, 1, 3, 2
        )  ## multiply the camera-to-world transformation matrices with the concatenated tensor (xyz) to get the projected coordinates of the points in the camera coordinate system (shape: (n, nv, 3, n_pts))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(
            -1
        )  ## compute the Euclidean norm of the projected coordinates along the last dimension and add a trailing singleton dimension (shape: (n, nv, 1, n_pts))
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(
            0, 1, 3, 2
        )  ## multiply the intrinsic camera matrices with the projected coordinates to get the pixel coordinates (shape: (n, nv, n_pts, 3) - 3rd dimension: x, y, and z coordinates in the pixel space)
        xy = xyz_projected[
            :, :, :, [0, 1]
        ]  ## select only the x and y coordinates of the pixel coordinates (shape: (n, nv, n_pts, 2))
        z_ = xyz_projected[
            :, :, :, 2:3
        ]  ## select only the z coordinate of the pixel coordinates (shape: (n, nv, n_pts, 1))

        # This scales the x-axis into the right range.
        xy = xy / z_.clamp_min(EPS)
        invalid = (
            (z_ <= EPS)  ## c.f. paper Handling invalid samples: if the depth exceeds a certrain threshold
            | (xy[:, :, :, :1] < -1)  ## Note that the color value range is [-1,1]
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )  ## Invalid points are points outside the image or points with invalid depth. This creates a boolean tensor of shape (n, nv, 1, n_pts), where each element is True if the corresponding point is invalid.

        sampled_colors = (
            F.grid_sample(
                self.grid_c_imgs.view(n_ * nv_, c_, h_, w_),
                xy.view(n_ * nv_, 1, -1, 2),
                mode=self.color_interpolation,
                padding_mode="border",
                align_corners=False,
            )
            .view(n_, nv_, c_, n_pts)
            .permute(0, 1, 3, 2)
        )  ## Sample colors from the grid using the projected world coordinates.

        assert not torch.any(
            torch.isnan(sampled_colors)
        )  ## Check that there are no NaN values in the sampled colors tensor.

        if (
            self.grid_c_combine is not None
        ):  ## If self.grid_c_combine is not None, combine colors from multiple points in the same group.
            invalid_groups, sampled_colors_groups = [], []

            for (
                group
            ) in (
                self.grid_c_combine
            ):  ## group:=list of indices that correspond to a subset of the total set of points in the point cloud. These subsets are combined to create a single image of the entire point cloud from multiple views.
                if (
                    len(group) == 1
                ):  ## If the group contains only one point, append the corresponding invalid tensor and sampled colors tensor to the respective lists.
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[
                    :, group
                ]  ## Otherwise, combine colors from the group by picking the color of the first valid point in the group.
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[
                    1
                ]  ## Get the index of the first valid point in the group.
                invalid_picked = torch.gather(
                    invalid_to_combine, dim=1, index=indices
                )  ## Pick the invalid tensor and sampled colors tensor corresponding to the first valid point in the group.
                colors_picked = torch.gather(
                    colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1])
                )

                invalid_groups.append(
                    invalid_picked
                )  ## Append the picked invalid tensor and sampled colors tensor to the respective lists.
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(
                invalid_groups, dim=1
            )  ## Concatenate the invalid tensors and sampled colors tensors along the second dimension.
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if (
            self.return_sample_depth
        ):  ## If self.return_sample_depth is True, concatenate the sample depth to the sampled colors tensor.
            distance = distance.view(n_, nv_, n_pts, 1)
            sampled_colors = torch.cat(
                (sampled_colors, distance), dim=-1
            )  ## cat along the last elem (c.f. paper pipeline)

        return sampled_colors, invalid  ## Return the sampled colors tensor and the invalid tensor.

    def compute_angle(self, xyz, query_pose, train_poses):  ## Assume batch size = 1
        """## code taken from projection.py in IBRNet
        :param xyz: [n_rays, n_samples, 3]        ### [n_rays, 64, 3]
        :param query_camera: [34, ]         | [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_cameras: [n_views, 34] | [1, n_views, 34]
        :return: ray_diff: [n_rays, n_samples, 4] |
        [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        # train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)                     # [n_views, 4, 4]
        num_views = len(train_poses)  ## ! Note: this should not be the batch size, but nv_
        # query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)  ## take translation x,y,z for pt
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6
        ray2train_pose = train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)
        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6

        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))  ## default: reshape((num_views, )
        return ray_diff

    def compute_ray_diff(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
        """
        :param xyz: [B, n_pts, 3]        ### [n_rays, 64, 3]
        :param view_dirs: [B, n_pts, 3]         | [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: ray_diff: [B, n_pts, 4] |
        [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        (
            B,
            n_pts,
            _,
        ) = xyz.shape  ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        B, nv_, c_, h_, w_ = self.grid_f_features[self._scale].shape  ### [B, nv, C, 192, 640]
        # if not use_single_featuremap:   nv_ = self.nv_
        xyz = xyz.unsqueeze(1)  # (B, 1, n_pts, 3)   ## to match grid_f_poses_w2c shape
        ones = torch.ones_like(
            xyz[..., :1]
        )  # Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  # (B, 1, n_pts, 4) Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = (self.grid_f_poses_w2c[:, :nv_, :3, :] @ xyz.permute(0, 1, 3, 2)).permute(
            0, 3, 1, 2
        )  # (B, n_pts, n_views, 3) Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-1, keepdim=True)  # (B, n_pts, n_views, 1)

        xyz_norm = xyz_projected / (distance + 1e-6)  # (B, n_views, n_pts, 3)
        view_dirs_norm = (view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)).unsqueeze(
            -2
        )  # (B, n_pts, 1, 3)

        ray_diff = xyz_norm - view_dirs_norm
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(xyz_norm * view_dirs_norm, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((B, n_pts, nv_, 4))  ## default: reshape((num_views, )
        return ray_diff

    def coords_to_voxel_grids(self, ref_coords, bev_h, bev_w, pc_range):
        ref_grids = copy.deepcopy(ref_coords)
        ref_grids[..., 0] = ((ref_grids[..., 0] - pc_range[0]) /
                            (pc_range[3] - pc_range[0])) * bev_w
        ref_grids[..., 1] = ((ref_grids[..., 1] - pc_range[1]) /
                            (pc_range[4] - pc_range[1])) * bev_h
        return ref_grids
    
    # TODO: source view with cor. query cam pose
    def forward(  ## Inference
        self, xyz, coarse=True, viewdirs=None, far=False, only_density=False, pgt=False, bev_plane = None
    ):  ## , infer=None):  ## ? "far"  ## Note: this forward propagation can be used for both training and eval
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B==sb, M//sb, 3) | e.g. [B_==4, M==8192, 3] | [n_rays, n_samples, n_views, n_feat] == img_feat
        B is batch of points (in rays)
        :param viewdirs split(viewdirs, eval_batch_size, dim=eval_batch_dim)
        :param infer when the feeded points are to be inferred. When training, we need to feed both viewdirs + xyz, so we need to specify if it is for training or inference.
        :return (B, 4) r g b sigma
        """
        """context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code."""
        with profiler.record_function(
            "model_inference"
        ):  ## create object with the name "model_inference". ## stop the timer when exiting the block
            n_, n_pts, _ = xyz.shape  ## n_ := Batch_size, n_pts == M
            # nv_ = 1
            nv_ = self.grid_c_imgs.shape[1]  ## 4 == (stereo 2 + side fish eye cam 2)
            ## Note that it's arbitrary offset for BEV, remember to properly set the offset accordingly.
            point_cloud_range = [-100, -50.0, -12.0, 250.0, 50.0, 12.0] # (x_back, y_right, z_bottom, x_front, y_left, z_top)
            x_back, y_right, z_bottom, x_front, y_left, z_top = [-100, -50.0, -12.0, 250.0, 50.0, 12.0]
            # grid_size = [
            #     int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),  # length # 4096
            #     int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),  # width # 1024
            #     int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])  # height # 50
            # ]

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, self.invalid_features, sampled_bottleneck_features = self.sample_features(
                # xyz, self.grid_t_pos, use_single_featuremap=False
                xyz,
                use_single_featuremap=False,
            )  # (B, n_pts, n_views, C), (B, n_pts, n_views), (B, n_pts, n_views, C_bottleneck)
            # sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"
            ### [1*batch_size, 4, Msb, 103]
            
            mlp_input = sampled_features.flatten(0, 1)  # (B * n_pts, n_views, C)
            
            if bev_plane:
                BEV = bev_plane
            else:
                print("__BEV not found, randomly initialize to test the BEVNet forward.")
                BEV = torch.rand(64, 64, 254, device=xyz.device)
            _, H_, W_ = BEV.shape

            # # 3D space range for BEV map in global coordinates
            # x_range = (-100, 250)  # x-axis range in meters
            # y_range = (-50, 50)    # y-axis range in meters
                        
            # # Scaling factors for x and y to map global coordinates to BEV grid
            # scale_x, scale_y = W_ / (x_range[1] - x_range[0]), H_ / (y_range[1] - y_range[0])

            # # Function to convert global coordinates (x, y) to BEV grid coordinates (u, v)
            # # g2bev = lambda x,y : ((x-x_range[0])*scale_x, (y-y_range[0])*scale_y)       # global_to_bev_grid
            # def global_to_bev_grid(x, y):
            #     # Scale and shift to BEV grid coordinates
            #     u = (x - x_range[0]) * scale_x
            #     v = (y - y_range[0]) * scale_y
            #     return u, v
            
            # viz purpose for pc
            pc = xyz.view(1, self.img_height, self.img_width, self.n_coarse, 3) 
            # from Real World coords to BEV coords - TODO: z is front facing, y is width, and x is height in BTS ray casting scheme
            uv = self.coords_to_voxel_grids(xyz[...,-2:], H_, W_, pc_range = point_cloud_range)
            # from BEV coords to -1, +1 for F.grid_sample
            # TODO: Check if the normalization is correct. May be the issue is with the order of operations.
            # uv[..., 0] = uv[..., 0] / H_ * 2 - 1
            # uv[..., 1] = uv[..., 1] / W_ * 2 - 1
            # uv[..., 0] = (uv[..., 0] - uv[..., 0].min()) / (uv[...,0].max() - uv[...,0].min()) * 2 - 1
            # uv[..., 1] = (uv[..., 1] - uv[..., 1].min()) / (uv[...,1].max() - uv[...,1].min()) * 2 - 1
            # Normalize x coordinates (u)
            uv[..., 0] = (uv[..., 0] - x_back) / (x_front - x_back) * 2 - 1
            # Normalize y coordinates (v)
            uv[..., 1] = (uv[..., 1] - y_right) / (y_left - y_right) * 2 - 1
            # TODO: Define width and Z from ray casting from the camera iamge such the BEV plane can be interpolated accordingly.
            # ray_sample_points = torch.stack([
            #     torch.linspace(-100, 250, n_samples),  # x-coordinates in 3D space
            #     torch.linspace(-50, 50, n_samples)     # y-coordinates in 3D space
            # ], dim=1)  # Shape: (n_samples, 2)
            
            # Convert each sampled point from global coordinates to BEV grid coordinates
            # bev_grid_points = g2bev(xyz[0,:,0], xyz[0,:,1])
            # bev_grid_points = torch.stack([g2bev(x, y) for x, y in xyz[0,:,:2]], dim=0)
            
            # # Remove the loop and process all  points at once
            # x, y = xyz[0, :, 0], xyz[0, :, 1]  # xyz.shape == [B, M, 3:xyz]
            
            # # Vectorized transformation
            # bev_x = (x - x_range[0]) * scale_x  
            # bev_y = (y - y_range[0]) * scale_y  
            
            # # Stack x,y coordinates 
            # bev_grid_points = torch.stack([bev_x, bev_y], dim=1)
            # bev_grid_points = bev_grid_points.unsqueeze(0).unsqueeze(2)  # Shape (1, n_samples, 1, 2)

            # # Normalize coordinates for grid_sample
            # bev_grid_points[:, :, :, 0] = (bev_grid_points[:, :, :, 0] / (W_ - 1)) * 2 - 1  # Normalize u
            # bev_grid_points[:, :, :, 1] = (bev_grid_points[:, :, :, 1] / (H_ - 1)) * 2 - 1  # Normalize v

            # TODO: Interpolate features at the sampled points using bilinear interpolation (_Renderer)
            interpolated_features = F.grid_sample(BEV.unsqueeze(0), uv.unsqueeze(0), 
                                                  mode='bilinear', padding_mode='zeros',
                                                  align_corners=True)  
            interpolated_features = interpolated_features.squeeze(0).squeeze(1) # Shape (C, n_samples)
            mlp_input = interpolated_features.permute(1,0).unsqueeze(1) # Shape (n_pts, n_views == 1, C)
            

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, self.invalid_features, sampled_bottleneck_features = self.sample_features(
                # xyz, self.grid_t_pos, use_single_featuremap=False
                xyz,
                use_single_featuremap=False,
            )  # (B, n_pts, n_views, C), (B, n_pts, n_views), (B, n_pts, n_views, C_bottleneck)
            # sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"
            ### [1*batch_size, 4, Msb, 103]

            # mlp_input = sampled_features.flatten(0, 1)  # (B * n_pts, n_views, C)

            # Camera frustum culling stuff, currently disabled
            combine_index, dim_size = None, None

            kwargs = {
                # "invalid_features": self.invalid_features.flatten(0, 1),  # (B* n_pts, n_views)
                "invalid_features": None,  # (B* n_pts, n_views)
                "combine_inner_dims": (n_pts,),
                "combine_index": combine_index,
                "dim_size": dim_size,
            }

            if sampled_bottleneck_features is not None:
                ray_diff = self.compute_ray_diff(xyz, viewdirs)  # (B, n_pts, n_views, 4)
                kwargs.update(
                    {
                        "bottleneck_feats": sampled_bottleneck_features.flatten(
                            0, 1
                        ),  # (B * n_pts, n_views, C_bottleneck)
                        "ray_diff": ray_diff.flatten(0, 1),  # (B * n_pts, n_views, 4)
                    }
                )

            head_outputs = {
                # name: head(mlp_input, **kwargs).reshape(n_, n_pts, self._d_out) for name, head in self.heads.items()
                name: head(mlp_input, **{**kwargs, "head_name": name}).reshape(n_, -1, 1)
                for name, head in self.heads.items()
            }

            # if self.decoder_heads_conf["freeze"]: head_outputs["multiviewhead"].requires_grad = False  ## This is already done in MVhead initialization network: knowledge distillation

            mlp_output = head_outputs[self.final_pred_head]

            if self.sample_color:
                sigma = mlp_output[
                    ..., :1
                ]  ## TODO: vs multiview_signma c.f. 265 nerf.py for single_view vs multi_view_sigma
                sigma = F.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)
                # rgb = interpolated_features
            else:  ## RGB colors and invalid colors are computed directly from the mlp_output tensor. i.e. w/o calling sample_colors(xyz)
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = interpolated_features[1:4, :].reshape(n_, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                if self.invalid_features is not None:
                    invalid_colors = self.invalid_features.unsqueeze(-2)
                else: invalid_colors = None
                nv_ = 1

            mlp_outputs = [head_outputs[name] for name in head_outputs]

            # if pgt:
            #     residual = mlp_outputs[0].detach() - mlp_outputs[1]          ### multiviewhead(GT) - singleviewhead(pred)
            #     numer = torch.sqrt(torch.pow(residual, 2).sum())    ## L2 norm
            #     denom_ = torch.tensor(mlp_outputs[0].shape).sum()   ## to normalize with the constant
            #     # denom_ = residual.max() - residual.min()    ## to normalize with the Min-Max range [0,1]
            #     loss_pgt = float((numer / denom_).cpu())                   ## Min-Max Normalization

            if self.empty_empty:  ## method sets the sigma values of the invalid features to 0 for invalidity.
                sigma[torch.all(self.invalid_features, dim=-1)] = 0  # sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu

            """Combine RGB colors and invalid colors"""
            if not only_density:
                _, _, _, c_ = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_ * c_)  # (n, pts, nv * 3)
                if invalid_colors is not None:
                    invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_)
                    invalid = (
                        invalid_colors | torch.all(self.invalid_features, dim=-1)[..., None]
                    )  # invalid = invalid_colors | torch.all(invalid_features, dim=1).expand(-1,-1,invalid_colors.shape[-1])       # # invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
                    invalid = invalid.to(rgb.dtype)
                else: invalid = None
            else:  ## If only_density is True, the method only returns the volume density (sigma) without computing the RGB colors.
                rgb = torch.zeros((n_, n_pts, nv_ * 3), device=sigma.device)
                invalid = self.invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma, None
        # return rgb, torch.prod(invalid, dim=-1), sigma


class MVBTSNet(torch.nn.Module):
    def __init__(
        self,
        conf,
        encoder: nn.Module,
        code_xyz,
        heads: Dict[str, nn.Module],
        final_pred_head: Optional[str] = None,
        ren_nc=None,
        code_token=None,
    ):  ## dependency injection
        super().__init__()
        ## model constructor's complexity
        self.encoder = encoder
        self.code_xyz = code_xyz
        self.code_token = code_token
        self.heads = nn.ModuleDict(heads)
        if final_pred_head:
            self.final_pred_head = final_pred_head
        else:
            self.final_pred_head = list(self.heads.keys())[0]  ### [('multiviewhead' | 'singleviewhead')][0] ## knowledge-distillation purpose

        self.requires_bottleneck_feats = False

        for _, head in self.heads.items():
            if hasattr(head, "require_bottleneck_feats"):
                if head.require_bottleneck_feats and (
                    head.independent_token_net.__class__.__name__ == "NeuRayIndependentToken"
                ):  ## For read out token type: "NeuRayIndependentToken"
                    self.requires_bottleneck_feats = True
                    break
        self.use_viewdirs = conf.get("use_viewdirs", True)

        self.n_coarse = ren_nc
        # self.decoder_heads_conf = conf.get("decoder_heads")
        # self.loss_pgt = conf.get("loss_pgt")

        self.d_min, self.d_max = conf.get("z_near"), conf.get("z_far")
        self.learn_empty, self.empty_empty, self.inv_z = (
            conf.get("learn_empty", True),
            conf.get("empty_empty", False),
            conf.get("inv_z", True),
        )
        self.color_interpolation, self.code_mode = conf.get("color_interpolation", "bilinear"), conf.get(
            "code_mode", "z"
        )
        if self.code_mode not in ["z", "distance", "z_feat"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        ## For potential AE model (from GeoNeRF)
        ## config to compute total sample convoluted, ts_conv
        # self.ts_conv = (
        #     (ren_nc // 4) * (conf.get("patch_size") * conf.get("patch_size")) * (conf.get("ray_batch_size") // ren_nc)
        # )
        # self.AE = conf.get("AE")
        # ts_ = batch_size * (self.n_coarse // 4) * self.patch_size self.patch_size * (ray_batch_size // self.n_coarse)

        # move outside of the class
        # self.encoder = make_backbone(conf["encoder"])  ### ResNetEncoder + Monodepth2 as decoder
        # self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)
        self.flip_augmentation = conf.get("flip_augmentation", False)
        self.return_sample_depth = conf.get("return_sample_depth", False)
        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out  ### 64 + 39
        d_out = 1 if self.sample_color else 4
        ## If sample_color is set to False, then d_out is set to 4 to represent the RGBA color values
        ## (red, green, blue, alpha) of the reconstructed scene. If sample_color is set to True, then d_out is set to 1
        ## to represent the estimated depth value of the reconstructed scene.

        self._d_in, self._d_out = d_in, d_out
        # self.mlp_coarse, self.mlp_fine = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out), make_mlp(
        #     conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True
        # )

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))
        ## factor to multiply the output of the corresponding MLP in the forward, which helps to control the range of the output values from the MLP
        self._scale = 0  ## set spatial resolution size accoridng to the scale of output feature map from the encoder
        self.invalid_features = None

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None, model_name=None):
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * 0.5 + 0.5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None
        ## Note: This is yet to be feature map before passing img to encoder
        n_, nv_, c_, h_, w_ = images_encoder.shape  ### [n_, nv_, 3:=RGB, 192, 640]
        c_l = self.encoder.latent_size  ### 64 c.f. paper D.1.

        if self.flip_augmentation and self.training:  ## data augmentation for color
            do_flip = (torch.rand(1) > 0.5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1,))

        if self.requires_bottleneck_feats:
            image_latents_ms, bottleneck_feats_ms = self.encoder(
                images_encoder.view(n_ * nv_, c_, h_, w_)
            )  ## Encoder of BTS's backbone model e.g. Monodepth2
        elif not self.requires_bottleneck_feats:
            image_latents_ms, _ = self.encoder(
                images_encoder.view(n_ * nv_, c_, h_, w_)
            )  ## Encoder of BTS's backbone model e.g. Monodepth2
            bottleneck_feats_ms = None
        else:
            NotImplementedError(f"__unrecognized input self.requires_bottleneck_feats:{self.requires_bottleneck_feats}")

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1,)) for il in image_latents_ms]
            if bottleneck_feats_ms is not None:
                bottleneck_feats_ms = [torch.flip(bf, dims=(-1,)) for bf in bottleneck_feats_ms]

        _, _, h_, w_ = image_latents_ms[
            0
        ].shape  ## get spatial resol from 1st layer out of 4 from feature maps generated by Enc
        image_latents_ms = [
            F.interpolate(image_latents, size=(h_, w_)).view(n_, nv_, c_l, h_, w_) for image_latents in image_latents_ms
        ]  ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer
        # img_feat_ms = [F.interpolate(feat_latents, size=(h_, w_)).view(n_, nv_, img_feat_ms[-1].shape[1], h_, w_) for feat_latents in img_feat_ms]    ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer
        if bottleneck_feats_ms is not None:
            self.grid_f_bottleneck_feats = F.interpolate(bottleneck_feats_ms[-1], size=(h_, w_)).view(
                n_, nv_, bottleneck_feats_ms[-1].shape[1], h_, w_
            )  ## upsampling the feature maps from down-sampled 4 layers to the same spatial resolution of 1st layer

        ## feature
        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        ## color
        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

        self.grid_t_pos = ids_encoder  ## positional embedding for token's order

    def sample_features(
        self,
        xyz,
        use_single_featuremap=True
        # self, xyz, grid_t_pos, use_single_featuremap=True
    ):  ## 2nd arg: to control whether multiple feature maps should be combined into a single feature map or not. If True, the function will average the sampled features from multiple feature maps along the view dimension (nv) before returning the result. This can be useful when you want to combine information from multiple views or feature maps into a single representation.
        (
            n_,
            n_pts,
            _,
        ) = xyz.shape  ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        n_, nv_, c_, h_, w_ = self.grid_f_features[self._scale].shape  ### [B, nv, C, 192, 640]
        # if not use_single_featuremap:   nv_ = self.nv_
        xyz = xyz.unsqueeze(
            1
        )  # (n, 1, pts, 3)                            ## Add a singleton dimension to the input point cloud to match grid_f_poses_w2c shape
        ones = torch.ones_like(
            xyz[..., :1]
        )  ## Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  ## Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = (self.grid_f_poses_w2c[:, :nv_, :3, :]) @ xyz.permute(
            0, 1, 3, 2
        )  ## Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)  ### [1, 2, 100000, 1]
        xyz_projected = (self.grid_f_Ks[:, :nv_] @ xyz_projected).permute(
            0, 1, 3, 2
        )  ## Apply the intrinsic camera parameters to the projected points to get pixel coordinates
        xy = xyz_projected[:, :, :, [0, 1]]  ## Extract the x,y coordinates and depth value from the projected points
        z_ = xyz_projected[:, :, :, 2:3]

        xy = xy / z_.clamp_min(
            EPS
        )  ## Normalize the x,y coordinates by the depth value and check for invalid points    => image coord -> pixel coord
        invalid = (
            (z_ <= EPS)
            | (xy[:, :, :, :1] < -1)
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )
        """given a vector p = (x, y, z) this is the difference of normalizing either:z ||p|| = sqrt(x^2 + y^2 + z^2). So you either give the network (x, y, z_normalized) or (x, y, ||p||_normalized) as input. It is just different parameterizations of the same point."""
        if (
            self.code_mode == "z"
        ):  ## Depending on the code mode, normalize the depth value or distance value to the [-1, 1] range and concatenate with the xy coordinates
            # Get z into [-1, 1] range  ## Normalizing the z coordinates leads to a consistent positional encoding of the viewing information. In line 172 the viewing information (xyz_projected) is given to a positional encoder before it is appended to the overall feature vector
            if self.inv_z:
                z_ = (1 / z_.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z_ = (z_ - self.d_min) / (self.d_max - self.d_min)
            z_ = 2 * z_ - 1
            xyz_projected = torch.cat((xy, z_), dim=-1)  ## concatenates the normalized x, y, and z coordinates

        elif self.code_mode == "z_feat":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1

            feat_map_pos_enc = torch.randn(distance.shape).to(distance.device)
            for i in range(distance.shape[1]):
                feat_map_pos_enc[:, i, :, :] = feat_map_pos_enc[:, i, :, :] + self.grid_t_pos[i]
            feat_map_pos_enc = nn.Parameter(feat_map_pos_enc)

            xyz_projected = torch.cat(
                (xy, distance, feat_map_pos_enc), dim=-1
            )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)

        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat(
                (xy, distance), dim=-1
            )  ## Apply the positional encoder to the concatenated xy and depth/distance coordinates (it enables the model to capture more complex spatial dependencies without a significant increase in model complexity or training data)
        xyz_code = (
            self.code_xyz(xyz_projected.view(n_ * nv_ * n_pts, -1)).view(n_, nv_, n_pts, -1).permute(0, 2, 1, 3)
        )  ## ! positional encoding dimension to check (concatenate)

        feature_map = self.grid_f_features[self._scale][:, :nv_]  ## !

        # These samples are from different scales
        if (
            self.learn_empty
        ):  ## "empty space" can refer to areas in a scene where there is no object, or it could also refer to areas that are not observed or are beyond the range of the sensor. This allows the model to have a distinct learned representation for "empty" space, which can be beneficial in tasks like 3D reconstruction where understanding both the objects in a scene and the empty space between them is important.
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c_).expand(
                n_, nv_, n_pts, c_
            )  ## trainable parameter, initialized with random features
        ## feature_map (2, 4, 64, 128, 128): n_ = 2, nv_ = 4 views, c_ = 64 channels in the feature map, and the height and width of the feature map are h = 128 and w = 128
        ## !TODO: for multiviews for F.grid_sample : xy.view(n_ * nv_, 1, -1, 2) To debug how xy looks like in order to integrate for multiview (by looking over doc in Pytorch regarding how to sample all frames)
        sampled_features = (
            F.grid_sample(
                feature_map.view(n_ * nv_, c_, h_, w_),
                xy.view(n_ * nv_, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .view(n_, nv_, c_, n_pts)
            .permute(0, 3, 1, 2)
        )  ## set x,y coordinates as grid feature

        if self.requires_bottleneck_feats:
            self.grid_f_bottleneck_feats = self.grid_f_bottleneck_feats[
                :, :nv_
            ]  ## taking last layer of encoder to feed into NeuRay
            sampled_bottleneck_feats = (
                F.grid_sample(
                    self.grid_f_bottleneck_feats.view(n_ * nv_, self.grid_f_bottleneck_feats.shape[2], h_, w_),
                    xy.view(n_ * nv_, 1, -1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                .view(n_, nv_, self.grid_f_bottleneck_feats.shape[2], n_pts)
                .permute(0, 3, 1, 2)
            )
        else:
            sampled_bottleneck_feats = None  ### dim(sampled_features): (n_, nv_, n_pts, c_)

        if (
            self.learn_empty
        ):  ## Replace invalid features in the sampled features tensor with the corresponding features from the expanded empty feature
            sampled_features[invalid.expand(-1, -1, -1, c_)] = empty_feature_expanded[
                invalid.expand(-1, -1, -1, c_)
            ]  ## broadcasting and make it fit to feature map
        ## dim(xyz): (B,M), M:=#_pts.

        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)  ### (n_, nv_, M, C1+C_pos_emb)
        # sampled_features = torch.cat((sampled_features, xyz_code, token_pos_enc), dim=-1)  ### (n_, nv_, M, C1+C_pos_emb)

        return (
            sampled_features,
            invalid[..., 0].permute(0, 2, 1),
            sampled_bottleneck_feats,
        )  ## output BTS's decoder   ### img_feat_sampled.shape == (n_, n_pts, nv_, C_)

    def sample_colors(self, xyz):
        n_, n_pts, _ = xyz.shape  ## n := batch size, n_pts := #_points in world coord.
        n_, nv_, c_, h_, w_ = self.grid_c_imgs.shape  ## nv_ := #_views
        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(
            xyz[..., :1]
        )  ## create a tensor of ones with the same shape as the first two dimensions of (xyz), up to the third dimension, and add a trailing singleton dimension (shape: (n, 1)) e.g. (n, 1, pts, 1)
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  ## concatenates the tensor of ones with xyz along the last dimension (i.e., the dimension representing the coordinates of each point).
        xyz_projected = (self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(
            0, 1, 3, 2
        )  ## multiply the camera-to-world transformation matrices with the concatenated tensor (xyz) to get the projected coordinates of the points in the camera coordinate system (shape: (n, nv, 3, n_pts))
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(
            -1
        )  ## compute the Euclidean norm of the projected coordinates along the last dimension and add a trailing singleton dimension (shape: (n, nv, 1, n_pts))
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(
            0, 1, 3, 2
        )  ## multiply the intrinsic camera matrices with the projected coordinates to get the pixel coordinates (shape: (n, nv, n_pts, 3) - 3rd dimension: x, y, and z coordinates in the pixel space)
        xy = xyz_projected[
            :, :, :, [0, 1]
        ]  ## select only the x and y coordinates of the pixel coordinates (shape: (n, nv, n_pts, 2))
        z_ = xyz_projected[
            :, :, :, 2:3
        ]  ## select only the z coordinate of the pixel coordinates (shape: (n, nv, n_pts, 1))

        # This scales the x-axis into the right range.
        xy = xy / z_.clamp_min(EPS)
        invalid = (
            (z_ <= EPS)  ## c.f. paper Handling invalid samples: if the depth exceeds a certrain threshold
            | (xy[:, :, :, :1] < -1)  ## Note that the color value range is [-1,1]
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )  ## Invalid points are points outside the image or points with invalid depth. This creates a boolean tensor of shape (n, nv, 1, n_pts), where each element is True if the corresponding point is invalid.

        sampled_colors = (
            F.grid_sample(
                self.grid_c_imgs.view(n_ * nv_, c_, h_, w_),
                xy.view(n_ * nv_, 1, -1, 2),
                mode=self.color_interpolation,
                padding_mode="border",
                align_corners=False,
            )
            .view(n_, nv_, c_, n_pts)
            .permute(0, 1, 3, 2)
        )  ## Sample colors from the grid using the projected world coordinates.

        assert not torch.any(
            torch.isnan(sampled_colors)
        )  ## Check that there are no NaN values in the sampled colors tensor.

        if (
            self.grid_c_combine is not None
        ):  ## If self.grid_c_combine is not None, combine colors from multiple points in the same group.
            invalid_groups, sampled_colors_groups = [], []

            for (
                group
            ) in (
                self.grid_c_combine
            ):  ## group:=list of indices that correspond to a subset of the total set of points in the point cloud. These subsets are combined to create a single image of the entire point cloud from multiple views.
                if (
                    len(group) == 1
                ):  ## If the group contains only one point, append the corresponding invalid tensor and sampled colors tensor to the respective lists.
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[
                    :, group
                ]  ## Otherwise, combine colors from the group by picking the color of the first valid point in the group.
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[
                    1
                ]  ## Get the index of the first valid point in the group.
                invalid_picked = torch.gather(
                    invalid_to_combine, dim=1, index=indices
                )  ## Pick the invalid tensor and sampled colors tensor corresponding to the first valid point in the group.
                colors_picked = torch.gather(
                    colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1])
                )

                invalid_groups.append(
                    invalid_picked
                )  ## Append the picked invalid tensor and sampled colors tensor to the respective lists.
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(
                invalid_groups, dim=1
            )  ## Concatenate the invalid tensors and sampled colors tensors along the second dimension.
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if (
            self.return_sample_depth
        ):  ## If self.return_sample_depth is True, concatenate the sample depth to the sampled colors tensor.
            distance = distance.view(n_, nv_, n_pts, 1)
            sampled_colors = torch.cat(
                (sampled_colors, distance), dim=-1
            )  ## cat along the last elem (c.f. paper pipeline)

        return sampled_colors, invalid  ## Return the sampled colors tensor and the invalid tensor.

    def compute_angle(self, xyz, query_pose, train_poses):  ## Assume batch size = 1
        """## code taken from projection.py in IBRNet
        :param xyz: [n_rays, n_samples, 3]        ### [n_rays, 64, 3]
        :param query_camera: [34, ]         | [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_cameras: [n_views, 34] | [1, n_views, 34]
        :return: ray_diff: [n_rays, n_samples, 4] |
        [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        # train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)                     # [n_views, 4, 4]
        num_views = len(train_poses)  ## ! Note: this should not be the batch size, but nv_
        # query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)  ## take translation x,y,z for pt
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6
        ray2train_pose = train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)
        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6

        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))  ## default: reshape((num_views, )
        return ray_diff

    def compute_ray_diff(self, xyz: torch.Tensor, view_dirs: torch.Tensor) -> torch.Tensor:
        """
        :param xyz: [B, n_pts, 3]        ### [n_rays, 64, 3]
        :param view_dirs: [B, n_pts, 3]         | [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: ray_diff: [B, n_pts, 4] |
        [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        (
            B,
            n_pts,
            _,
        ) = xyz.shape  ## Get the shape of the input point cloud and the feature grid (n, pts, spatial_coordinate == 3)
        B, nv_, c_, h_, w_ = self.grid_f_features[self._scale].shape  ### [B, nv, C, 192, 640]
        # if not use_single_featuremap:   nv_ = self.nv_
        xyz = xyz.unsqueeze(1)  # (B, 1, n_pts, 3)   ## to match grid_f_poses_w2c shape
        ones = torch.ones_like(
            xyz[..., :1]
        )  # Create a tensor of ones to add a fourth dimension to the point cloud for homogeneous coordinates
        xyz = torch.cat(
            (xyz, ones), dim=-1
        )  # (B, 1, n_pts, 4) Concatenate the tensor of ones with the point cloud to create homogeneous coordinates
        xyz_projected = (self.grid_f_poses_w2c[:, :nv_, :3, :] @ xyz.permute(0, 1, 3, 2)).permute(
            0, 3, 1, 2
        )  # (B, n_pts, n_views, 3) Apply the camera poses to the point cloud to get the projected points and calculate the distance
        distance = torch.norm(xyz_projected, dim=-1, keepdim=True)  # (B, n_pts, n_views, 1)

        xyz_norm = xyz_projected / (distance + 1e-6)  # (B, n_views, n_pts, 3)
        view_dirs_norm = (view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)).unsqueeze(
            -2
        )  # (B, n_pts, 1, 3)

        ray_diff = xyz_norm - view_dirs_norm
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(xyz_norm * view_dirs_norm, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((B, n_pts, nv_, 4))  ## default: reshape((num_views, )
        return ray_diff

    ## TODO: source view with cor. query cam pose
    def forward(  ## Inference
        self, xyz, coarse=True, viewdirs=None, far=False, only_density=False, pgt=False
    ):  ## , infer=None):  ## ? "far"  ## Note: this forward propagation can be used for both training and eval
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B==sb, M//sb, 3) | e.g. [B_==4, M==8192, 3] | [n_rays, n_samples, n_views, n_feat] == img_feat
        B is batch of points (in rays)
        :param viewdirs split(viewdirs, eval_batch_size, dim=eval_batch_dim)
        :param infer when the feeded points are to be inferred. When training, we need to feed both viewdirs + xyz, so we need to specify if it is for training or inference.
        :return (B, 4) r g b sigma
        """
        """context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code."""
        with profiler.record_function(
            "model_inference"
        ):  ## create object with the name "model_inference". ## stop the timer when exiting the block
            n_, n_pts, _ = xyz.shape  ## n_ := Batch_size, n_pts == M
            nv_ = self.grid_c_imgs.shape[1]  ## 4 == (stereo 2 + side fish eye cam 2)

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, self.invalid_features, sampled_bottleneck_features = self.sample_features(
                # xyz, self.grid_t_pos, use_single_featuremap=False
                xyz,
                use_single_featuremap=False,
            )  # (B, n_pts, n_views, C), (B, n_pts, n_views), (B, n_pts, n_views, C_bottleneck)
            # sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"
            ### [1*batch_size, 4, Msb, 103]

            mlp_input = sampled_features.flatten(0, 1)  # (B * n_pts, n_views, C)

            # Camera frustum culling stuff, currently disabled
            combine_index, dim_size = None, None

            kwargs = {
                "invalid_features": self.invalid_features.flatten(0, 1),  # (B* n_pts, n_views)
                "combine_inner_dims": (n_pts,),
                "combine_index": combine_index,
                "dim_size": dim_size,
            }

            if sampled_bottleneck_features is not None:
                ray_diff = self.compute_ray_diff(xyz, viewdirs)  # (B, n_pts, n_views, 4)
                kwargs.update(
                    {
                        "bottleneck_feats": sampled_bottleneck_features.flatten(
                            0, 1
                        ),  # (B * n_pts, n_views, C_bottleneck)
                        "ray_diff": ray_diff.flatten(0, 1),  # (B * n_pts, n_views, 4)
                    }
                )

            head_outputs = {
                # name: head(mlp_input, **kwargs).reshape(n_, n_pts, self._d_out) for name, head in self.heads.items()
                name: head(mlp_input, **{**kwargs, "head_name": name}).reshape(n_, -1, self._d_out)
                for name, head in self.heads.items()
            }

            # if self.decoder_heads_conf["freeze"]: head_outputs["multiviewhead"].requires_grad = False  ## This is already done in MVhead initialization network: knowledge distillation

            mlp_output = head_outputs[self.final_pred_head]

            if self.sample_color:
                sigma = mlp_output[
                    ..., :1
                ]  ## TODO: vs multiview_signma c.f. 265 nerf.py for single_view vs multi_view_sigma
                sigma = F.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)
            else:  ## RGB colors and invalid colors are computed directly from the mlp_output tensor. i.e. w/o calling sample_colors(xyz)
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n_, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                invalid_colors = self.invalid_features.unsqueeze(-2)
                nv_ = 1

            mlp_outputs = [head_outputs[name] for name in head_outputs]

            # if pgt:
            #     residual = mlp_outputs[0].detach() - mlp_outputs[1]          ### multiviewhead(GT) - singleviewhead(pred)
            #     numer = torch.sqrt(torch.pow(residual, 2).sum())    ## L2 norm
            #     denom_ = torch.tensor(mlp_outputs[0].shape).sum()   ## to normalize with the constant
            #     # denom_ = residual.max() - residual.min()    ## to normalize with the Min-Max range [0,1]
            #     loss_pgt = float((numer / denom_).cpu())                   ## Min-Max Normalization

            if self.empty_empty:  ## method sets the sigma values of the invalid features to 0 for invalidity.
                sigma[torch.all(self.invalid_features, dim=-1)] = 0  # sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu

            """Combine RGB colors and invalid colors"""
            if not only_density:
                _, _, _, c_ = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_ * c_)  # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_)

                invalid = (
                    invalid_colors | torch.all(self.invalid_features, dim=-1)[..., None]
                )  # invalid = invalid_colors | torch.all(invalid_features, dim=1).expand(-1,-1,invalid_colors.shape[-1])       # # invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:  ## If only_density is True, the method only returns the volume density (sigma) without computing the RGB colors.
                rgb = torch.zeros((n_, n_pts, nv_ * 3), device=sigma.device)
                invalid = self.invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma, None
        # return rgb, torch.prod(invalid, dim=-1), sigma


class MVBTSNet2(MVBTSNet):
    def __init__(
        self,
        conf,
        encoder: nn.Module,
        code_xyz,
        heads: Dict[str, nn.Module],
        final_pred_head: str,
        ren_nc=None,
        code_token=None,
    ):  ## dependency injection
        super().__init__(
            conf,
            encoder,
            code_xyz,
            {"standard_head": make_mlp(conf["mlp_fine"], 1, 1, True)},
            "standard_head",
            ren_nc,
            code_token,
        )

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)
        self.mlp_confidence = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        self.sigma_fusion = conf.get("sigma_fusion", True)

    def forward(  ## Inference
        self, xyz, coarse=True, viewdirs=None, far=False, only_density=False, pgt=False
    ):  ## , infer=None):  ## ? "far"  ## Note: this forward propagation can be used for both training and eval
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B==sb, M//sb, 3) | e.g. [B_==4, M==8192, 3] | [n_rays, n_samples, n_views, n_feat] == img_feat
        B is batch of points (in rays)
        :param viewdirs split(viewdirs, eval_batch_size, dim=eval_batch_dim)
        :param infer when the feeded points are to be inferred. When training, we need to feed both viewdirs + xyz, so we need to specify if it is for training or inference.
        :return (B, 4) r g b sigma
        """
        """context manager that helps to measure the execution time of the code block inside it. i.e. used to profile the execution time of the forward pass of the model during inference for performance analysis and optimization purposes. ## to analyze the performance of the code block, helping developers identify bottlenecks and optimize their code."""
        with profiler.record_function(
            "model_inference"
        ):  ## create object with the name "model_inference". ## stop the timer when exiting the block
            n_, n_pts, _ = xyz.shape  ## n_ := Batch_size, n_pts == M
            nv_ = self.grid_c_imgs.shape[1]  ## 4 == (stereo 2 + side fish eye cam 2)

            if self.grid_c_combine is not None:
                nv_ = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]   ## c + xyz_code := combined dimensionality of the features and the positional encoding c.f. (paper) Fig.2
            sampled_features, invalid_features, sampled_bottleneck_features = self.sample_features(
                # xyz, self.grid_t_pos, use_single_featuremap=False
                xyz,
                use_single_featuremap=False,
            )  # (B, n_pts, n_views, C), (B, n_pts, n_views), (B, n_pts, n_views, C_bottleneck)
            # sampled_features = sampled_features.reshape(n * n_pts, -1)  ## n_pts := number of points per "ray"
            ### [1*batch_size, 4, Msb, 103]

            mlp_input = sampled_features.flatten(0, 1)  # (B * n_pts, n_views, C)

            # Camera frustum culling stuff, currently disabled
            combine_index, dim_size = None, None

            # Run main NeRF network
            mlp_output = self.mlp_coarse(
                mlp_input,
                combine_inner_dims=(n_pts,),
                combine_index=combine_index,
                dim_size=dim_size,
            )
            conf_output = self.mlp_confidence(
                mlp_input,
                combine_inner_dims=(n_pts,),
                combine_index=combine_index,
                dim_size=dim_size,
            )

            weights = torch.nn.functional.softmax(
                conf_output[..., 0].masked_fill(invalid_features.flatten(0, 1) == 1, -1e9), dim=-1
            )
            sigmas = F.softplus(mlp_output)
            if self.sigma_fusion:
                sigma = torch.sum(sigmas * weights.unsqueeze(-1), dim=-2)
            else:
                sigma = F.softplus(torch.sum(mlp_output * weights.unsqueeze(-1), dim=-2))

            rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv_, pts, 3)

            """Combine RGB colors and invalid colors"""
            if not only_density:
                _, _, _, c_ = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_ * c_)  # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n_, n_pts, nv_)

                invalid = (
                    invalid_colors | torch.all(invalid_features, dim=-1)[..., None]
                )  # invalid = invalid_colors | torch.all(invalid_features, dim=1).expand(-1,-1,invalid_colors.shape[-1])       # # invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:  ## If only_density is True, the method only returns the volume density (sigma) without computing the RGB colors.
                rgb = torch.zeros((n_, n_pts, nv_ * 3), device=sigma.device)
                invalid = self.invalid_features.to(sigma.dtype)

            state_dict = {
                "final_sigma": sigma[None],
                "invalid_features": invalid_features.flatten(0, 1)[None],
                "sigmas": sigmas[None],
            }
        return (
            rgb,
            invalid,
            sigma[None],
            state_dict,
        )  # B*n_ptsx3, B*n_ptsx1, B*n_ptsx1, # B*n_ptsx... (B*n_ptsxn_viewsx...)


class BTSNet(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.d_min = conf.get("z_near")
        self.d_max = conf.get("z_far")

        self.learn_empty = conf.get("learn_empty", True)
        self.empty_empty = conf.get("empty_empty", False)
        self.inv_z = conf.get("inv_z", True)

        self.color_interpolation = conf.get("color_interpolation", "bilinear")
        self.code_mode = conf.get("code_mode", "z")
        if self.code_mode not in ["z", "distance"]:
            raise NotImplementedError(f"Unknown mode for positional encoding: {self.code_mode}")

        self.encoder = make_backbone(conf["encoder"])
        self.code_xyz = PositionalEncoding.from_conf(conf["code"], d_in=3)

        self.flip_augmentation = conf.get("flip_augmentation", False)

        self.return_sample_depth = conf.get("return_sample_depth", False)

        self.sample_color = conf.get("sample_color", True)

        d_in = self.encoder.latent_size + self.code_xyz.d_out
        d_out = 1 if self.sample_color else 4

        self._d_in = d_in
        self._d_out = d_out

        ## default
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_out=d_out)
        self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, d_out=d_out, allow_empty=True)

        ## to compatible with MVBTS simultaneously
        # kwargs = {
        #         "d_out": d_out,
        #         "allow_empty": True,
        #     }
        # self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, kwargs)
        # self.mlp_fine = make_mlp(conf["mlp_fine"], d_in, kwargs)

        if self.learn_empty:
            self.empty_feature = nn.Parameter(torch.randn((self.encoder.latent_size,), requires_grad=True))

        self._scale = 0

    def set_scale(self, scale):
        self._scale = scale

    def get_scale(self):
        return self._scale

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(self, images, Ks, poses_c2w, ids_encoder=None, ids_render=None, images_alt=None, combine_ids=None, model_name=None):
        poses_w2c = torch.inverse(poses_c2w)

        if ids_encoder is None:
            images_encoder = images
            Ks_encoder = Ks
            poses_w2c_encoder = poses_w2c
            ids_encoder = list(range(len(images)))
        else:
            images_encoder = images[:, ids_encoder]
            Ks_encoder = Ks[:, ids_encoder]
            poses_w2c_encoder = poses_w2c[:, ids_encoder]

        if images_alt is not None:
            images = images_alt
        else:
            images = images * 0.5 + 0.5

        if ids_render is None:
            images_render = images
            Ks_render = Ks
            poses_w2c_render = poses_w2c
            ids_render = list(range(len(images)))
        else:
            images_render = images[:, ids_render]
            Ks_render = Ks[:, ids_render]
            poses_w2c_render = poses_w2c[:, ids_render]

        if combine_ids is not None:
            combine_ids = list(list(group) for group in combine_ids)
            get_combined = set(sum(combine_ids, []))
            for i in range(images.shape[1]):
                if i not in get_combined:
                    combine_ids.append((i,))
            remap_encoder = {v: i for i, v in enumerate(ids_encoder)}
            remap_render = {v: i for i, v in enumerate(ids_render)}
            comb_encoder = [[remap_encoder[i] for i in group if i in ids_encoder] for group in combine_ids]
            comb_render = [[remap_render[i] for i in group if i in ids_render] for group in combine_ids]
            comb_encoder = [group for group in comb_encoder if len(group) > 0]
            comb_render = [group for group in comb_render if len(group) > 0]
        else:
            comb_encoder = None
            comb_render = None

        n, nv, c, h, w = images_encoder.shape
        c_l = self.encoder.latent_size

        if self.flip_augmentation and self.training:
            do_flip = (torch.rand(1) > 0.5).item()
        else:
            do_flip = False

        if do_flip:
            images_encoder = torch.flip(images_encoder, dims=(-1,))

        image_latents_ms = self.encoder(images_encoder.view(n * nv, c, h, w))
        if model_name != "pixelnerf": image_latents_ms = image_latents_ms[0]
        # if model_name == "pixelnerf":
        #     image_latents_ms = self.encoder(images_encoder.view(n * nv, c, h, w))
        # else:
        #     image_latents_ms, _ = self.encoder(images_encoder.view(n * nv, c, h, w))

        if do_flip:
            image_latents_ms = [torch.flip(il, dims=(-1,)) for il in image_latents_ms]

        _, _, h_, w_ = image_latents_ms[0].shape
        image_latents_ms = [
            F.interpolate(image_latents, (h_, w_)).view(n, nv, c_l, h_, w_) for image_latents in image_latents_ms
        ]

        self.grid_f_features = image_latents_ms
        self.grid_f_Ks = Ks_encoder
        self.grid_f_poses_w2c = poses_w2c_encoder
        self.grid_f_combine = comb_encoder

        self.grid_c_imgs = images_render
        self.grid_c_Ks = Ks_render
        self.grid_c_poses_w2c = poses_w2c_render
        self.grid_c_combine = comb_render

    def sample_features(self, xyz, use_single_featuremap=True):
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_f_features[self._scale].shape

        # if use_single_featuremap:
        #     nv = 1

        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1)
        xyz_projected = (self.grid_f_poses_w2c[:, :nv, :3, :]) @ xyz.permute(0, 1, 3, 2)
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)
        xyz_projected = (self.grid_f_Ks[:, :nv] @ xyz_projected).permute(0, 1, 3, 2)
        xy = xyz_projected[:, :, :, [0, 1]]
        z = xyz_projected[:, :, :, 2:3]

        xy = xy / z.clamp_min(EPS)
        invalid = (
            (z <= EPS)
            | (xy[:, :, :, :1] < -1)
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )

        if self.code_mode == "z":
            # Get z into [-1, 1] range
            if self.inv_z:
                z = (1 / z.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                z = (z - self.d_min) / (self.d_max - self.d_min)
            z = 2 * z - 1
            xyz_projected = torch.cat((xy, z), dim=-1)
        elif self.code_mode == "distance":
            if self.inv_z:
                distance = (1 / distance.clamp_min(EPS) - 1 / self.d_max) / (1 / self.d_min - 1 / self.d_max)
            else:
                distance = (distance - self.d_min) / (self.d_max - self.d_min)
            distance = 2 * distance - 1
            xyz_projected = torch.cat((xy, distance), dim=-1)
        xyz_code = self.code_xyz(xyz_projected.view(n * nv * n_pts, -1)).view(n, nv, n_pts, -1)

        feature_map = self.grid_f_features[self._scale][:, :nv]
        # These samples are from different scales
        if self.learn_empty:
            empty_feature_expanded = self.empty_feature.view(1, 1, 1, c).expand(n, nv, n_pts, c)

        sampled_features = (
            F.grid_sample(
                feature_map.view(n * nv, c, h, w),
                xy.view(n * nv, 1, -1, 2),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .view(n, nv, c, n_pts)
            .permute(0, 1, 3, 2)
        )

        if self.learn_empty:
            sampled_features[invalid.expand(-1, -1, -1, c)] = empty_feature_expanded[invalid.expand(-1, -1, -1, c)]

        sampled_features = torch.cat((sampled_features, xyz_code), dim=-1)

        # If there are multiple frames with predictions, reduce them.
        # TODO: Technically, this implementations should be improved if we use multiple frames.
        # The reduction should only happen after we perform the unprojection.

        if self.grid_f_combine is not None:
            invalid_groups = []
            sampled_features_groups = []

            for group in self.grid_f_combine:
                if len(group) == 1:
                    invalid_groups.append(invalid[:, group])
                    sampled_features_groups.append(sampled_features[:, group])

                invalid_to_combine = invalid[:, group]
                features_to_combine = sampled_features[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)
                features_picked = torch.gather(
                    features_to_combine, dim=1, index=indices.expand(-1, -1, -1, features_to_combine.shape[-1])
                )

                invalid_groups.append(invalid_picked)
                sampled_features_groups.append(features_picked)

            invalid = torch.cat(invalid_groups, dim=1)
            sampled_features = torch.cat(sampled_features_groups, dim=1)

        if use_single_featuremap:
            sampled_features = sampled_features.mean(dim=1)
            invalid = torch.any(invalid, dim=1)

        return sampled_features, invalid

    def sample_colors(self, xyz):
        n, n_pts, _ = xyz.shape
        n, nv, c, h, w = self.grid_c_imgs.shape
        xyz = xyz.unsqueeze(1)  # (n, 1, pts, 3)
        ones = torch.ones_like(xyz[..., :1])
        xyz = torch.cat((xyz, ones), dim=-1)
        xyz_projected = (self.grid_c_poses_w2c[:, :, :3, :]) @ xyz.permute(0, 1, 3, 2)
        distance = torch.norm(xyz_projected, dim=-2).unsqueeze(-1)
        xyz_projected = (self.grid_c_Ks @ xyz_projected).permute(0, 1, 3, 2)
        xy = xyz_projected[:, :, :, [0, 1]]
        z = xyz_projected[:, :, :, 2:3]

        # This scales the x-axis into the right range.
        xy = xy / z.clamp_min(EPS)
        invalid = (
            (z <= EPS)
            | (xy[:, :, :, :1] < -1)
            | (xy[:, :, :, :1] > 1)
            | (xy[:, :, :, 1:2] < -1)
            | (xy[:, :, :, 1:2] > 1)
        )

        sampled_colors = (
            F.grid_sample(
                self.grid_c_imgs.view(n * nv, c, h, w),
                xy.view(n * nv, 1, -1, 2),
                mode=self.color_interpolation,
                padding_mode="border",
                align_corners=False,
            )
            .view(n, nv, c, n_pts)
            .permute(0, 1, 3, 2)
        )
        assert not torch.any(torch.isnan(sampled_colors))

        if self.grid_c_combine is not None:
            invalid_groups = []
            sampled_colors_groups = []

            for group in self.grid_c_combine:
                if len(group) == 1:
                    invalid_groups.append(invalid[:, group])
                    sampled_colors_groups.append(sampled_colors[:, group])
                    continue

                invalid_to_combine = invalid[:, group]
                colors_to_combine = sampled_colors[:, group]

                indices = torch.min(invalid_to_combine, dim=1, keepdim=True)[1]
                invalid_picked = torch.gather(invalid_to_combine, dim=1, index=indices)
                colors_picked = torch.gather(
                    colors_to_combine, dim=1, index=indices.expand(-1, -1, -1, colors_to_combine.shape[-1])
                )

                invalid_groups.append(invalid_picked)
                sampled_colors_groups.append(colors_picked)

            invalid = torch.cat(invalid_groups, dim=1)
            sampled_colors = torch.cat(sampled_colors_groups, dim=1)

        if self.return_sample_depth:
            distance = distance.view(n, nv, n_pts, 1)
            sampled_colors = torch.cat((sampled_colors, distance), dim=-1)

        return sampled_colors, invalid

    def forward(self, xyz, coarse=True, viewdirs=None, far=False, only_density=False):
        """
        Predict (r, g, b, sigma) at world space points xyz.
        Please call encode first!
        :param xyz (B, 3)
        B is batch of points (in rays)
        :return (B, 4) r g b sigma
        """

        with profiler.record_function("model_inference"):
            n, n_pts, _ = xyz.shape
            nv = self.grid_c_imgs.shape[1]

            if self.grid_c_combine is not None:
                nv = len(self.grid_c_combine)

            # Sampled features all has shape: scales [n, n_pts, c + xyz_code]
            sampled_features, invalid_features = self.sample_features(
                xyz, use_single_featuremap=not only_density
            )  # invalid features (n, n_pts, 1)
            sampled_features = sampled_features.reshape(n * n_pts, -1)

            mlp_input = sampled_features.view(n, n_pts, -1)

            # Camera frustum culling stuff, currently disabled
            combine_index = None
            dim_size = None

            # Run main NeRF network
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(n_pts,),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            # (n, pts, c) -> (n, n_pts, c)
            mlp_output = mlp_output.reshape(n, n_pts, self._d_out)

            if self.sample_color:
                sigma = mlp_output[..., :1]
                sigma = F.softplus(sigma)
                rgb, invalid_colors = self.sample_colors(xyz)  # (n, nv, pts, 3)
            else:
                sigma = mlp_output[..., :1]
                sigma = F.relu(sigma)
                rgb = mlp_output[..., 1:4].reshape(n, 1, n_pts, 3)
                rgb = F.sigmoid(rgb)
                invalid_colors = invalid_features.unsqueeze(-2)
                nv = 1

            if self.empty_empty:
                sigma[invalid_features[..., 0]] = 0
            # TODO: Think about this!
            # Since we don't train the colors directly, lets use softplus instead of relu

            if not only_density:
                _, _, _, c = rgb.shape
                rgb = rgb.permute(0, 2, 1, 3).reshape(n, n_pts, nv * c)  # (n, pts, nv * 3)
                invalid_colors = invalid_colors.permute(0, 2, 1, 3).reshape(n, n_pts, nv)

                invalid = invalid_colors | invalid_features  # Invalid features gets broadcasted to (n, n_pts, nv)
                invalid = invalid.to(rgb.dtype)
            else:
                rgb = torch.zeros((n, n_pts, nv * 3), device=sigma.device)
                invalid = invalid_features.to(sigma.dtype)
        return rgb, invalid, sigma
