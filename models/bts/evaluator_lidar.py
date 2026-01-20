import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.bts.model.models_bts import MVBTSNet  ## BTSNet
from models.bts.model.ray_sampler import ImageRaySampler
from utils.base_evaluator import base_evaluation
from utils.metrics import MeanMetric
from utils.projection_operations import distance_to_z


from datasets.data_util import make_datasets
from datasets.kitti_odom.kitti_odometry_dataset import KittiOdometryDataset
from datasets.kitti_raw.kitti_raw_dataset import KittiRawDataset
from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.head_util import make_head
from models.common.model.scheduler import make_scheduler
from models.common.render import NeRFRenderer
from models.bts.model.image_processor import make_image_processor, RGBProcessor
from models.bts.model.loss import ReconstructionLoss, compute_errors_l1ssim
from models.bts.model.models_bts import MVBTSNet  ## default: BTSNet
from models.bts.model.ray_sampler import (
    ImageRaySampler,
    PatchRaySampler,
    RandomRaySampler,
)
from models.ibrnet.config import config_parser
from models.ibrnet.ibrwrapper import IBRNetRenderingWrapper
from models.ibrnet.model import IBRNetModel
from models.ibrnet.projection import Projector
from scripts.inference_setup import render_profile
from utils.base_trainer import base_training
from utils.metrics import MeanMetric
from utils.plotting import color_tensor
from utils.projection_operations import distance_to_z

from torch import profiler
from torchvision.utils import make_grid

IDX = 0
EPS = 1e-4

# The KITTI 360 cameras have a 5 degrees negative inclination. We need to account for that.
cam_incl_adjust = torch.tensor(
    [
        [1.0000000, 0.0000000, 0.0000000, 0],
        [0.0000000, 0.9961947, 0.0871557, 0],
        [0.0000000, -0.0871557, 0.9961947, 0],
        [0.0000000, 000000000, 0.0000000, 1],
    ],
    dtype=torch.float32,
).view(1, 1, 4, 4)


def get_pts(x_range, y_range, z_range, ppm, ppm_y, y_res=None):  ## ppm:=pts_per_meter
    x_res = abs(int((x_range[1] - x_range[0]) * ppm))
    if y_res is None:
        y_res = abs(int((y_range[1] - y_range[0]) * ppm_y))
    z_res = abs(int((z_range[1] - z_range[0]) * ppm))
    x = (
        torch.linspace(x_range[0], x_range[1], x_res)
        .view(1, 1, x_res)
        .expand(y_res, z_res, -1)
    )
    z = (
        torch.linspace(z_range[0], z_range[1], z_res)
        .view(1, z_res, 1)
        .expand(y_res, -1, x_res)
    )
    if y_res == 1:
        y = (
            torch.tensor([y_range[0] * 0.5 + y_range[1] * 0.5])
            .view(y_res, 1, 1)
            .expand(-1, z_res, x_res)
        )
    else:
        y = (
            torch.linspace(y_range[0], y_range[1], y_res)
            .view(y_res, 1, 1)
            .expand(-1, z_res, x_res)
        )
    xyz = torch.stack((x, y, z), dim=-1)

    return xyz, (x_res, y_res, z_res)


# This function takes all points between min_y and max_y and projects them into the x-z plane.
# To avoid cases where there are no points at the top end, we consider also points that are beyond the maximum z distance.
# The points are then converted to polar coordinates and sorted by angle.


def get_lidar_slices(point_clouds, velo_poses, y_range, y_res, max_dist):
    slices = []
    ys = torch.linspace(y_range[0], y_range[1], y_res)
    if y_res > 1:
        slice_height = ys[1] - ys[0]
    else:
        slice_height = 0
    n_bins = 360

    for y in ys:
        if y_res == 1:
            min_y = y
            max_y = y_range[-1]
        else:
            min_y = y - slice_height / 2
            max_y = y + slice_height / 2

        slice = []

        for pc, velo_pose in zip(point_clouds, velo_poses):
            pc_world = (velo_pose @ pc.T).T

            mask = ((pc_world[:, 1] >= min_y) & (pc_world[:, 1] <= max_y)) | (
                torch.norm(pc_world[:, :3], dim=-1) >= max_dist
            )

            slice_points = pc[mask, :2]

            angles = torch.atan2(slice_points[:, 1], slice_points[:, 0])
            dists = torch.norm(slice_points, dim=-1)

            slice_points_polar = torch.stack((angles, dists), dim=1)
            # Sort by angles for fast lookup
            slice_points_polar = slice_points_polar[torch.sort(angles)[1], :]

            slice_points_polar_binned = torch.zeros_like(slice_points_polar[:n_bins, :])
            bin_borders = torch.linspace(
                -math.pi, math.pi, n_bins + 1, device=slice_points_polar.device
            )

            dist = slice_points_polar[0, 1]

            # To reduce noise, we bin the lidar points into bins of 1deg and then take the minimum distance per bin.
            border_is = torch.searchsorted(slice_points_polar[:, 0], bin_borders)

            for i in range(n_bins):
                left_i, right_i = border_is[i], border_is[i + 1]
                angle = (bin_borders[i] + bin_borders[i + 1]) * 0.5
                if right_i > left_i:
                    dist = torch.min(slice_points_polar[left_i:right_i, 1])
                slice_points_polar_binned[i, 0] = angle
                slice_points_polar_binned[i, 1] = dist

            slice_points_polar = slice_points_polar_binned

            # Append first element to last to have full 360deg coverage
            slice_points_polar = torch.cat(
                (
                    torch.tensor(
                        [
                            [
                                slice_points_polar[-1, 0] - math.pi * 2,
                                slice_points_polar[-1, 1],
                            ]
                        ],
                        device=slice_points_polar.device,
                    ),
                    slice_points_polar,
                    torch.tensor(
                        [
                            [
                                slice_points_polar[0, 0] + math.pi * 2,
                                slice_points_polar[0, 1],
                            ]
                        ],
                        device=slice_points_polar.device,
                    ),
                ),
                dim=0,
            )

            slice.append(slice_points_polar)

        slices.append(slice)

    return slices


def check_occupancy(pts, slices, velo_poses, min_dist=3):
    is_occupied = torch.ones_like(pts[:, 0])
    is_visible = torch.zeros_like(pts[:, 0], dtype=torch.bool)

    thresh = (len(slices[0]) - 2) / len(slices[0])

    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)

    world_to_velos = torch.inverse(velo_poses)

    step = pts.shape[0] // len(slices)

    for i, slice in enumerate(slices):
        for j, (lidar_polar, world_to_velo) in enumerate(zip(slice, world_to_velos)):
            pts_velo = (world_to_velo @ pts[i * step : (i + 1) * step, :].T).T

            # Convert query points to polar coordinates in velo space
            angles = torch.atan2(pts_velo[:, 1], pts_velo[:, 0])
            dists = torch.norm(pts_velo, dim=-1)

            indices = torch.searchsorted(lidar_polar[:, 0].contiguous(), angles)

            left_angles = lidar_polar[indices - 1, 0]
            right_angles = lidar_polar[indices, 0]

            left_dists = lidar_polar[indices - 1, 1]
            right_dists = lidar_polar[indices, 1]

            interp = (angles - left_angles) / (right_angles - left_angles)
            surface_dist = left_dists * (1 - interp) + right_dists * interp

            is_occupied_velo = (dists > surface_dist) | (dists < min_dist)

            is_occupied[i * step : (i + 1) * step] += is_occupied_velo.float()

            if j == 0:
                is_visible[i * step : (i + 1) * step] |= ~is_occupied_velo

    is_occupied /= len(slices[0])

    is_occupied = is_occupied > thresh

    return is_occupied, is_visible


def project_into_cam(pts, proj, pose):
    pts = torch.cat((pts, torch.ones_like(pts[:, :1])), dim=-1)
    cam_pts = (proj @ (torch.inverse(pose).squeeze()[:3, :] @ pts.T)).T
    cam_pts[:, :2] /= cam_pts[:, 2:3]
    dist = cam_pts[:, 2]
    return cam_pts, dist


def plot(pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()

    rows = math.ceil(yd / 2)
    fig, axs = plt.subplots(rows, 2)

    for y in range(yd):
        r = y // 2
        c = y % 2

        if rows > 1:
            axs[r][c].imshow(pts[y], interpolation="none")
        else:
            axs[c].imshow(pts[y], interpolation="none")
    plt.show()


def plot_sperical(polar_pts):
    polar_pts = polar_pts.cpu()
    angles = polar_pts[:, 0]
    dists = polar_pts[:, 1]

    max_dist = dists.mean() * 2
    dists = dists.clamp(0, max_dist) / max_dist

    x = -torch.sin(angles) * dists
    y = torch.cos(angles) * dists

    plt.plot(x, y)
    plt.show()


def save(name, pts, xd, yd, zd):
    pts = pts.reshape(yd, zd, xd).cpu().numpy()[0]
    plt.imsave(name, pts)


def save_all(f, is_occupied, is_occupied_pred, images, xd, yd, zd):
    save(f"{f}_gt.png", is_occupied, xd, yd, zd)
    save(f"{f}_pred.png", is_occupied_pred, xd, yd, zd)
    plt.imsave(
        f"{f}_input.png", images[0, 0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5
    )


class BTSWrapper(nn.Module):
    def __init__(self, renderer, config, dataset) -> None:
        super().__init__()

        self.renderer = renderer

        self.renderer = renderer
        self.encoder_ids = config.get("encoder_ids", [0])
        self.save_bin_path = config.get("save_bin_path", None)

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.query_batch_size = config.get("query_batch_size", 50000)
        self.occ_threshold = 0.5

        ## c.f. BTS paper Tab 1.
        occ_pred = True
        if occ_pred:
            old = True
            if old:
                self.x_range = (-4, 4)
                self.y_range = (0, 0.75)
                self.z_range = (20, 4)
                self.ppm = 10
                self.ppm_y = 4
                self.y_res = 1
            else:
                self.x_range = (-4, 4)
                self.y_range = (1.0 / 3.0, 1.0)
                self.z_range = (20, 4)
                self.ppm = 10
                self.ppm_y = 2
                self.y_res = 3
        else:
            self.x_range = (-12, 12)
            self.y_range = (-1, 2)
            self.z_range = (20, 4)
            self.ppm = 10
            self.ppm_y = self.ppm

            self.y_res = None
        print(
            self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y, self.y_res
        )

        self.sampler = ImageRaySampler(self.z_near, self.z_far, channels=3)

        self.dataset = dataset
        self.aggregate_timesteps = 20
        self.counter = 0

        self.z_correct = []
        self.total = []

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def forward(self, data):
        if isinstance(self.renderer.net, IBRNetRenderingWrapper):
            n_samples = 64
            self.renderer.net.model.net_coarse.pos_encoding = (
                self.renderer.net.model.net_coarse.posenc(d_hid=16, n_samples=n_samples)
            )
            self.renderer.net.model.args.N_samples = n_samples
            self.renderer.net.regular_grid = False
        data = dict(data)
        images = torch.stack(data["imgs"], dim=1)  # n, v, c, h, w
        poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
        projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)
        index = data["index"].item()

        seq, id, is_right = self.dataset._datapoints[index]
        seq_len = self.dataset._img_ids[seq].shape[0]

        n, v, c, h, w = images.shape
        device = images.device

        T_velo_to_pose = torch.tensor(
            self.dataset._calibs["T_velo_to_pose"], device=device
        )

        # Our coordinate system is at the same position as cam0, but rotated 5deg up along the x axis to adjust for camera inclination.
        # Consequently, the xz plane is parallel to the street.
        world_transform = torch.inverse(poses[:, :1, :, :])
        world_transform = cam_incl_adjust.to(device) @ world_transform
        poses = world_transform @ poses

        self.sampler.height, self.sampler.width = h, w

        # Load lidar pointclouds
        points_all, velo_poses = [], []
        for id in range(id, min(id + self.aggregate_timesteps, seq_len)):
            points = np.fromfile(
                os.path.join(
                    self.dataset.data_path,
                    "data_3d_raw",
                    seq,
                    "velodyne_points",
                    "data",
                    f"{self.dataset._img_ids[seq][id]:010d}.bin",
                ),
                dtype=np.float32,
            ).reshape(-1, 4)
            points[:, 3] = 1.0
            points = torch.tensor(points, device=device)
            velo_pose = (
                world_transform.squeeze()
                @ torch.tensor(self.dataset._poses[seq][id], device=device)
                @ T_velo_to_pose
            )
            points_all.append(points)
            velo_poses.append(velo_pose)

        velo_poses = torch.stack(velo_poses, dim=0)

        rays, _ = self.sampler.sample(None, poses[:, :1, :, :], projs[:, :1, :, :])

        ## TODO: Making possible view combination cases for evaluation
        # enc2eval = [i for i in self.encoder_ids]
        # enc2eval = {
        #     mono: [0]                        ## monocular fixed case
        #     # mono_tmp: [0, 1]                     ## temporal mono case
        #     stereo: [0, 2]                     ## stereo fixed case
        #     # stereo_tmp: [0, 1, 2, 3]               ## stereo temporal case
        #     encoder_ids: [0, 1, 2, 3, 4, 5, 6, 7]   ## full case
        # }

        ids_encoder = self.encoder_ids
        self.renderer.net.compute_grid_transforms(
            projs[:, ids_encoder], poses[:, ids_encoder]
        )
        self.renderer.net.encode(
            images,
            projs,
            poses,
            ids_encoder=ids_encoder,
            ids_render=ids_encoder[:1],
            images_alt=images * 0.5 + 0.5,
        )
        self.renderer.net.set_scale(0)
        render_dict = self.renderer(rays, want_weights=True, want_alphas=True)
        if "fine" not in render_dict:
            render_dict["fine"] = dict(render_dict["coarse"])
        render_dict = self.sampler.reconstruct(render_dict)
        pred_depth = distance_to_z(render_dict["coarse"]["depth"], projs[:1, :1])

        # Get pts
        q_pts, (xd, yd, zd) = get_pts(
            self.x_range, self.y_range, self.z_range, self.ppm, self.ppm_y, self.y_res
        )
        # q_pts = q_pts.permute(0, 2, 1, 3).contiguous()
        q_pts_shape = q_pts.shape[:3]
        pts_shape = q_pts.shape
        q_pts = q_pts.to(images.device).view(-1, 3)

        # if isinstance(self.renderer.net, IBRNetRenderingWrapper):
        #     n_samples = 160
        #     self.renderer.net.model.net_coarse.pos_encoding = (
        #         self.renderer.net.model.net_coarse.posenc(d_hid=16, n_samples=n_samples)
        #     )
        #     self.renderer.net.model.args.N_samples = n_samples
        #     self.renderer.net.regular_grid = True

        # is visible? Check whether point is closer than the computed pseudo depth
        cam_pts, dists = project_into_cam(q_pts, projs[0, 0], poses[0, 0])
        pred_dist = F.grid_sample(
            pred_depth.view(1, 1, h, w),
            cam_pts[:, :2].view(1, 1, -1, 2),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).view(-1)
        is_visible_pred = dists <= pred_dist

        depth_plus4meters = False
        if depth_plus4meters:
            mask = (dists >= pred_dist) & (dists < pred_dist + 4)
            densities = torch.zeros_like(q_pts[..., 0])
            densities[mask] = 1.0
            is_occupied_pred = densities > self.occ_threshold
        else:
            # Query the density of the query points from the density field
            densities = []
            for i_from in range(0, len(q_pts), self.query_batch_size):
                i_to = min(i_from + self.query_batch_size, len(q_pts))
                q_pts_ = q_pts[i_from:i_to]
                _, _, densities_, _ = self.renderer.net(
                    q_pts_.unsqueeze(0), only_density=True
                )  ## ! occupancy estimation
                densities.append(densities_.squeeze(0))
            densities = torch.cat(densities, dim=0).squeeze()
            is_occupied_pred = densities > self.occ_threshold

        def pack(uncompressed):
            """convert a boolean array into a bitwise array."""
            uncompressed_r = uncompressed.reshape(-1, 8)
            compressed = uncompressed_r.dot(
                1 << np.arange(uncompressed_r.shape[-1] - 1, -1, -1)
            )
            return compressed

        if self.save_bin_path:
            # base_file = "/storage/user/hank/methods_test/semantic-kitti-api/bts_test/sequences/00/voxels"
            outside_frustum = (
                (
                    (cam_pts[:, 0] < -1.0)
                    | (cam_pts[:, 0] > 1.0)
                    | (cam_pts[:, 1] < -1.0)
                    | (cam_pts[:, 0] > 1.0)
                )
                .reshape(q_pts_shape)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            is_occupied_numpy = (
                is_occupied_pred.reshape(q_pts_shape)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            is_occupied_numpy[outside_frustum] = 0.0
            ## carving out the invisible regions out of view-frustum
            # for i_ in range(
            #     (is_occupied_numpy.shape[0]) // 2
            # ):  ## left | right half of the space
            #     for j_ in range(i_ + 1):
            #         is_occupied_numpy[i_, j_] = 0

            pack(np.flip(is_occupied_numpy, (0, 1, 2)).reshape(-1)).astype(
                np.uint8
            ).tofile(
                # f"{base_file}/{self.counter:0>6}.bin"
                f"{self.save_bin_path}/{self.counter:0>6}.bin"
            )
            # for idx_i, image in enumerate(images[0]):
            #     torchvision.utils.save_image(
            #         image, f"{self.save_bin_path}/{self.counter:0>6}_{idx_i}.png"
            #     )
            self.counter += 1

        # is occupied?
        slices = get_lidar_slices(
            points_all,
            velo_poses,
            self.y_range,
            yd,
            (self.z_range[0] ** 2 + self.x_range[0] ** 2) ** 0.5,
        )
        is_occupied, is_visible = check_occupancy(q_pts, slices, velo_poses)

        is_visible |= is_visible_pred

        # Only not visible points can be occupied
        is_occupied &= ~is_visible

        is_occupied_reshape = is_occupied.view(pts_shape[:-1])
        is_occupied_pred_reshape = is_occupied_pred.view(pts_shape[:-1])

        self.z_correct.append(
            (is_occupied_pred_reshape == is_occupied_reshape)
            .float()
            .sum(dim=0)
            .sum(dim=-1)
        )
        self.total.append(
            (is_occupied_pred_reshape == is_occupied_pred_reshape)
            .float()
            .sum(dim=0)
            .sum(dim=-1)
        )
        torch.save(torch.stack(self.z_correct, dim=0), "occupancy/z_correct.pt")
        torch.save(torch.stack(self.total, dim=0), "occupancy/total.pt")

        # is_occupied_acc = (
        #     (is_occupied_pred == is_occupied).float().mean().nan_to_num_(0.0).item()
        # )
        # is_occupied_prec = (
        #     is_occupied[is_occupied_pred].float().mean().nan_to_num_(0.0).item()
        # )
        # is_occupied_rec = (
        #     is_occupied_pred[is_occupied].float().mean().nan_to_num_(0.0).item()
        # )

        # not_occupied_not_visible_ratio = (
        #     ((~is_occupied) & (~is_visible)).float().mean().nan_to_num_(0.0).item()
        # )

        # total_ie = (
        #     ((~is_occupied) & (~is_visible)).float().sum().nan_to_num_(0.0).item()
        # )

        # ie_acc = (
        #     (is_occupied_pred == is_occupied)[(~is_visible)]
        #     .float()
        #     .mean()
        #     .nan_to_num_(0.0)
        #     .item()
        # )
        # ie_prec = (
        #     (~is_occupied)[(~is_occupied_pred) & (~is_visible)]
        #     .float()
        #     .mean()
        #     .nan_to_num_(0.0)
        #     .item()
        # )  ## ? why without .item()?
        # ie_rec = (
        #     (~is_occupied_pred)[(~is_occupied) & (~is_visible)]
        #     .float()
        #     .mean()
        #     .nan_to_num_(0.0)
        #     .item()
        # )
        is_occupied_acc = (is_occupied_pred == is_occupied).float().mean().item()
        is_occupied_prec = is_occupied[is_occupied_pred].float().mean().item()
        is_occupied_rec = is_occupied_pred[is_occupied].float().mean().item()

        not_occupied_not_visible_ratio = (
            ((~is_occupied) & (~is_visible)).float().mean().item()
        )

        total_ie = ((~is_occupied) & (~is_visible)).float().sum().item()

        ie_acc = (is_occupied_pred == is_occupied)[(~is_visible)].float().mean().item()
        ie_prec = (
            (~is_occupied)[(~is_occupied_pred) & (~is_visible)].float().mean().item()
        )  ## ? why without .item()?
        ie_rec = (
            (~is_occupied_pred)[(~is_occupied) & (~is_visible)].float().mean().item()
        )
        total_no_nop_nv = (
            ((~is_occupied) & (~is_occupied_pred))[(~is_visible) & (~is_occupied)]
            .float()
            .sum()
        )

        data["o_acc"] = is_occupied_acc
        data["o_rec"] = is_occupied_rec
        data["o_prec"] = is_occupied_prec
        data["ie_acc"] = ie_acc
        data["ie_rec"] = ie_rec
        data["ie_prec"] = ie_prec
        data["ie_r"] = not_occupied_not_visible_ratio
        data["t_ie"] = total_ie
        data["t_no_nop_nv"] = total_no_nop_nv

        data["z_near"] = torch.tensor(self.z_near, device=images.device)
        data["z_far"] = torch.tensor(self.z_far, device=images.device)

        ## updating for viz and eval metrics in Tensorboard
        data.update(self.compute_occ_metrics(data))
        globals()["IDX"] += 1

        return data

    def compute_occ_metrics(self, data):
        metrics_dict = {
            "o_acc": data["o_acc"],
            "o_rec": data["o_rec"],
            "o_prec": data["o_prec"],
            "ie_acc": data["ie_acc"],
            "ie_rec": data["ie_rec"],
            "ie_prec": data["ie_prec"],
            "ie_r": data["ie_r"],
            "t_ie": data["t_ie"],
            "t_no_nop_nv": data["t_no_nop_nv"],
        }

        return metrics_dict


def evaluation(local_rank, config):
    return base_evaluation(
        local_rank, config, get_dataflow, initialize, get_metrics, visualize
    )
    # return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


# def get_dataflow(config, logger=None):
def get_dataflow(config, logger=None):
    test_dataset = make_test_dataset(config["data"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    return test_loader


def get_metrics(config, device):
    names = [
        "o_acc",
        "o_prec",
        "o_rec",
        "ie_acc",
        "ie_prec",
        "ie_rec",
        "t_ie",
        "t_no_nop_nv",
    ]
    metrics = {
        name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device)
        for name in names
    }
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")
    # arch = config["model_conf"].get("arch", "BTSNet")
    sample_color = config["model_conf"].get("sample_color", True)
    d_out = 1 if sample_color else 4

    if arch == "MVBTSNet":
        encoder = make_backbone(config["model_conf"]["encoder"])
        code_xyz = PositionalEncoding.from_conf(config["model_conf"]["code"], d_in=3)

        d_in = encoder.latent_size + code_xyz.d_out  ### 103

        # Make configurable for Semantics as well
        decoder_heads = {
            head_conf["name"]: make_head(head_conf, d_in, d_out)
            for head_conf in config["model_conf"]["decoder_heads"]
        }

        # net = globals()[arch](config["model_conf"])
        # net = globals()[arch](config["model_conf"], ren_nc=config["renderer"]["n_coarse"], B_=config["batch_size"]) ## default: globals()[arch](config["model_conf"])
        net = MVBTSNet(
            config["model_conf"],
            encoder,
            code_xyz,
            decoder_heads,
            final_pred_head=config.get("final_prediction_head", None),
            ren_nc=config["renderer"]["n_coarse"],
        )
    elif arch == "IBRNet":
        parser = config_parser()
        args = parser.parse_known_args(parser._default_config_files)[0]
        args.ckpt_path = "/storage/user/hank/methods_test/IBRNet/out/pretraining/finished_kitti360_FeTrue_NoOffset_2023-11-06_11-15-43/model_250000.pth"
        args.N_samples = 64
        if config["data"].type == "KITTI_360" or "KITTI_360_DFT":
            args.num_source_views = 7
        else:
            args.num_source_views = 3
        # elif config['data'].type == "KITTI_"
        model = IBRNetModel(args, load_opt=False, load_scheduler=False)
        projector = Projector(device="cuda")
        net = IBRNetRenderingWrapper(model=model, projector=projector)

    elif arch == "neuray":
        pass

    else:
        net = globals()[arch](config["model_conf"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(
        renderer, config["model_conf"], make_test_dataset(config["data"])
    )

    return model


## TODO: viz metrics loss graph + figures at the end of evaluation step
def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    print(f"__Visualizing step:{step}")

    data = engine.state.output["output"]
    writer = logger.writer

    images = torch.stack(data["imgs"], dim=1).detach()[0]
    recon_imgs = data["fine"][0]["rgb"].detach()[0]
    recon_depths = [f["depth"].detach()[0] for f in data["fine"]]

    # depth_profile = data["coarse"][0]["weights"].detach()[0]
    depth_profile = data["coarse"][0]["alphas"].detach()[0]
    alphas = data["coarse"][0]["alphas"].detach()[0]
    invalids = data["coarse"][0]["invalid"].detach()[0]

    z_near = data["z_near"]
    z_far = data["z_far"]

    take_n = min(
        images.shape[0], 8
    )  ## num eval viz images to be shown in tensorboard, default: 6

    _, c, h, w = images.shape
    nv = recon_imgs.shape[0]

    images = images[:take_n]
    images = images * 0.5 + 0.5

    recon_imgs = recon_imgs.view(nv, h, w, -1, c)
    recon_imgs = recon_imgs[:take_n]
    # Aggregate recon_imgs by taking the mean
    recon_imgs = recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)

    recon_mse = (((images - recon_imgs) ** 2) / 2).mean(dim=1).clamp(0, 1)
    recon_mse = color_tensor(recon_mse, cmap="plasma").permute(0, 3, 1, 2)

    recon_depths = [
        (1 / d[:take_n] - 1 / z_far) / (1 / z_near - 1 / z_far) for d in recon_depths
    ]
    recon_depths = [
        color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2)
        for d in recon_depths
    ]

    depth_profile = (
        depth_profile[:take_n][:, [h // 4, h // 2, 3 * h // 4], :, :]
        .view(take_n * 3, w, -1)
        .permute(0, 2, 1)
    )
    depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
    depth_profile = color_tensor(depth_profile, cmap="plasma").permute(0, 3, 1, 2)

    alphas = alphas[:take_n]

    alphas += 1e-5

    ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
    ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (
        math.log2(alphas.shape[-1])
    )
    ray_entropy = color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)

    alpha_sum = (alphas.sum(dim=-1) / alphas.shape[-1]).clamp(-1)
    alpha_sum = color_tensor(alpha_sum, cmap="plasma").permute(0, 3, 1, 2)

    invalids = invalids[:take_n]
    invalids = invalids.mean(-2).mean(-1)
    invalids = color_tensor(invalids, cmap="plasma").permute(0, 3, 1, 2)

    # Write images
    nrow = int(take_n**0.5)

    # profile plotting
    profiles = torch.stack(data["profiles"], dim=0)
    profiles = color_tensor(profiles, cmap="magma", norm=True)
    profiles_grid = make_grid(profiles).permute(2, 0, 1)  ## Bird-eye view
    # TODO: provide GT LiDAR for bird-eye view for the comparison
    images_grid = make_grid(images, nrow=nrow)
    recon_imgs_grid = make_grid(recon_imgs, nrow=nrow)
    recon_depths_grid = [make_grid(d, nrow=nrow) for d in recon_depths]
    depth_profile_grid = make_grid(depth_profile, nrow=nrow)
    ray_entropy_grid = make_grid(ray_entropy, nrow=nrow)
    alpha_sum_grid = make_grid(alpha_sum, nrow=nrow)
    recon_mse_grid = make_grid(recon_mse, nrow=nrow)
    invalids_grid = make_grid(invalids, nrow=nrow)

    writer.add_image(
        f"{tag}/profiles", profiles_grid.cpu(), global_step=step
    )  ## Bird-eye view
    writer.add_image(f"{tag}/input_im", images_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_im", recon_imgs_grid.cpu(), global_step=step)
    for i, d in enumerate(recon_depths_grid):
        writer.add_image(f"{tag}/recon_depth_{i}", d.cpu(), global_step=step)
    writer.add_image(f"{tag}/depth_profile", depth_profile_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/ray_entropy", ray_entropy_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/alpha_sum", alpha_sum_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_mse", recon_mse_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/invalids", invalids_grid.cpu(), global_step=step)
