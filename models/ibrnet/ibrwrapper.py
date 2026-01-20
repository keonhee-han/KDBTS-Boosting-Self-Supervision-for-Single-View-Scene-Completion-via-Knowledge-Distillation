import torch
import torch.nn as nn

from models.ibrnet.model import IBRNetModel
from models.ibrnet.projection import Projector


class IBRNetRenderingWrapper(nn.Module):
    def __init__(
        self,
        model: IBRNetModel,
        projector: Projector,
        #  feature_net
    ) -> None:
        super().__init__()

        # self.feature_net = feature_net
        self.projector = projector
        self.model = model
        self.regular_grid = False

    def convert_camera(self, Ks, poses_c2w, ids, height, width):
        padded_Ks = torch.nn.functional.pad(
            Ks, (0, 1, 0, 1), mode="constant", value=0.0
        )
        # padded_Ks[..., 0:1, 2] = (padded_Ks[..., 0:1, 2] + 1.0) / 2.0
        # padded_Ks[..., 0, :] = padded_Ks[..., 0, :] * width
        # padded_Ks[..., 1, :] = padded_Ks[..., 1, :] * height
        padded_Ks[..., 3, 3] = 1.0
        cameras = torch.cat(
            [
                torch.ones_like(padded_Ks)[..., 0:1, 0] * height,
                torch.ones_like(padded_Ks)[..., 0:1, 0] * width,
                padded_Ks.flatten(-2, -1),
                # torch.linalg.inv(poses_c2w).flatten(-2, -1),
                poses_c2w.flatten(-2, -1),
            ],
            dim=-1,
        )

        return cameras[:, ids]

    def set_scale(self, scale):
        pass

    def convert_output(self, raw, pixel_mask):
        rgb = raw[:, :, :3].flatten(0, 1)[None]
        sigma = raw[:, :, 3].flatten(0, 1)[None]
        invalids = (~pixel_mask).flatten(0, 1)[None]
        return rgb, invalids, sigma, None

    def compute_grid_transforms(self, *args, **kwargs):
        pass

    def encode(
        self,
        images,
        Ks,
        poses_c2w,
        ids_encoder=None,
        ids_render=None,
        images_alt=None,
        combine_ids=None,
    ):
        ids_encoder = [1, 2, 3]
        ids_render = [0]  # TODO: check if correct

        self.featmaps = self.model.feature_net(images[:, ids_encoder].flatten(0, 1))

        self.src_imgs = images[:, ids_encoder].permute(0, 1, 3, 4, 2)
        height, width = images.shape[-2], images.shape[-1]
        self.src_cameras = self.convert_camera(
            Ks, poses_c2w, ids_encoder, height, width
        )
        self.que_camera = self.convert_camera(
            Ks, poses_c2w, ids_render, height, width
        ).flatten(0, 1)

    def forward(
        self, xyz, coarse=True, viewdirs=None, far=False, only_density=False, pgt=False
    ):
        que_camera = self.que_camera.clone()
        if self.regular_grid:
            que_camera_pose = torch.eye(4, device=self.que_camera.device)
            que_camera_pose[2, 3] = -10000.0        ##
            que_camera[:, -16:] = que_camera_pose.flatten(0, 1)[None]
        xyz = xyz.reshape(-1, self.model.args.N_samples, 3)
        rgb_feat, ray_diff, mask = self.projector.compute(
            xyz,
            que_camera,
            self.src_imgs,
            self.src_cameras,
            featmaps=self.featmaps[0],
        )
        pixel_mask = mask[..., 0].sum(dim=2) > 1  # TODO: maybe 0
        raw_coarse = self.model.net_coarse(rgb_feat, ray_diff, mask)

        return self.convert_output(raw_coarse, pixel_mask)
