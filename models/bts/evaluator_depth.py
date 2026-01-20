import math
from pathlib import Path

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import lpips
import skimage.metrics

from datasets.data_util import make_test_dataset
from models.common.render import NeRFRenderer
from models.bts.model.image_processor import make_image_processor, RGBProcessor
from models.bts.model.loss import ReconstructionLoss
from models.bts.model.models_bts import MVBTSNet  ## BTSNet
from models.bts.model.ray_sampler import (
    ImageRaySampler,
    PatchRaySampler,
    RandomRaySampler,
)
from models.ibrnet.config import config_parser
from models.ibrnet.ibrwrapper import IBRNetRenderingWrapper
from models.ibrnet.model import IBRNetModel
from models.ibrnet.projection import Projector
from utils.base_evaluator import base_evaluation
from utils.metrics import HistogramMetric, MeanMetric
from utils.projection_operations import distance_to_z

from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.head_util import make_head


IDX = 0


class BTSWrapper(nn.Module):
    def __init__(
        self,
        renderer,
        config,
    ) -> None:
        super().__init__()

        self.renderer = renderer

        self.z_near = config["z_near"]
        self.z_far = config["z_far"]
        self.ray_batch_size = config["ray_batch_size"]
        self.sampler = ImageRaySampler(self.z_near, self.z_far)

        self.load_from_plate = config.get("load_from_plate", False)
        self.depth_dir = config.get("depth_dir", None)
        if self.load_from_plate:
            self.counter = 0

        self.lpips_vgg = lpips.LPIPS(net="vgg")

        self.depth_scaling = config.get("depth_scaling", None)

        self.ids_enc_viz_eval = config.get("ids_enc_offset_viz", [0])  ##

    @staticmethod
    def get_loss_metric_names():
        return ["loss", "loss_l2", "loss_mask", "loss_temporal"]

    def load_depth_map(self, idx):
        depth_map = torch.load(Path(self.depth_dir).joinpath(f"{idx}_depth_gray.pt"))
        return depth_map

    def forward(self, data):
        data = dict(data)
        if self.load_from_plate:
            data["fine"] = [
                {
                    "depth": self.load_depth_map(self.counter)[None, None].to(
                        data["imgs"][0].device
                    )
                }
            ]  # 1, 1, 1, H, W
            self.counter = self.counter + 1
        else:
            images = torch.stack(data["imgs"], dim=1)  # n, v, c, h, w
            poses = torch.stack(data["poses"], dim=1)  # n, v, 4, 4 w2c
            projs = torch.stack(data["projs"], dim=1)  # n, v, 4, 4 (-1, 1)

            n, v, c, h, w = images.shape
            device = images.device

            # Use first frame as keyframe
            to_base_pose = torch.inverse(poses[:, :1, :, :])
            poses = to_base_pose.expand(-1, v, -1, -1) @ poses

            ids_encoder = self.ids_enc_viz_eval  ## fixed during eval
            ids_renderer = [0]

            self.renderer.net.compute_grid_transforms(
                projs[:, ids_encoder], poses[:, ids_encoder]
            )
            self.renderer.net.encode(
                images,
                projs,
                poses,
                ids_encoder=ids_encoder,
                ids_render=ids_renderer,
            )

            all_rays, all_rgb_gt = self.sampler.sample(images * 0.5 + 0.5, poses, projs)

            data["fine"] = []
            data["coarse"] = []

            self.renderer.net.set_scale(0)
            render_dict = self.renderer(all_rays, want_weights=True, want_alphas=True)

            if "fine" not in render_dict:
                render_dict["fine"] = dict(render_dict["coarse"])

            render_dict["rgb_gt"] = all_rgb_gt
            render_dict["rays"] = all_rays

            render_dict = self.sampler.reconstruct(render_dict)

            render_dict["coarse"]["depth"] = distance_to_z(
                render_dict["coarse"]["depth"], projs
            )
            render_dict["fine"]["depth"] = distance_to_z(
                render_dict["fine"]["depth"], projs
            )

            data["fine"].append(render_dict["fine"])
            data["coarse"].append(render_dict["coarse"])
            data["rgb_gt"] = render_dict["rgb_gt"]
            data["rays"] = render_dict["rays"]

            data["z_near"] = torch.tensor(self.z_near, device=images.device)
            data["z_far"] = torch.tensor(self.z_far, device=images.device)

        depth_metrics = self.compute_depth_metrics(data)
        rel_err = depth_metrics.pop("rel_err")
        abs_err = depth_metrics.pop("abs_err")
        threshold = depth_metrics.pop("threshold")

        # data.update(self.compute_depth_metrics(data))
        data.update(depth_metrics)

        data["abs_err"] = abs_err
        data["rel_err"] = rel_err
        data["threshold"] = threshold
        # data.update(self.compute_nvs_metrics(data))

        globals()["IDX"] += 1

        return data

    def compute_depth_metrics(self, data):
        # TODO: This is only correct for batchsize 1!
        depth_gt = data["depths"][0]
        depth_pred = data["fine"][0]["depth"][:, :1]

        torch.save(data["imgs"][0], f"depth/imgs/{globals()['IDX']}.pt")
        torch.save(depth_gt, f"depth/ground_truth/{globals()['IDX']}.pt")
        torch.save(depth_pred, f"depth/pred/{globals()['IDX']}.pt")

        depth_pred = F.interpolate(depth_pred, depth_gt.shape[-2:])

        if self.depth_scaling == "median":
            mask = depth_gt > 0
            scaling = torch.median(depth_gt[mask]) / torch.median(depth_pred[mask])
            depth_pred = scaling * depth_pred
        elif self.depth_scaling == "l2":
            mask = depth_gt > 0
            depth_pred = depth_pred
            depth_gt_ = depth_gt[mask]
            depth_pred_ = depth_pred[mask]
            depth_pred_ = torch.stack(
                (depth_pred_, torch.ones_like(depth_pred_)), dim=-1
            )
            x = torch.linalg.lstsq(
                depth_pred_.to(torch.float32), depth_gt_.unsqueeze(-1).to(torch.float32)
            ).solution.squeeze()
            depth_pred = depth_pred * x[0] + x[1]

        depth_pred = torch.clamp(depth_pred, 1e-3, 80)
        mask = depth_gt != 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        thresh = torch.maximum((depth_gt / depth_pred), (depth_pred / depth_gt))
        a1 = (thresh < 1.25).to(torch.float)
        a2 = (thresh < 1.25**2).to(torch.float)
        a3 = (thresh < 1.25**3).to(torch.float)
        a1 = a1.mean()
        a2 = a2.mean()
        a3 = a3.mean()

        rmse = (depth_gt - depth_pred) ** 2
        rmse = rmse.mean() ** 0.5

        rmse_log = (torch.log(depth_gt) - torch.log(depth_pred)) ** 2
        rmse_log = rmse_log.mean() ** 0.5

        abs_rel = torch.abs(depth_gt - depth_pred) / depth_gt
        abs_rel = abs_rel.mean()

        sq_rel = ((depth_gt - depth_pred) ** 2) / depth_gt
        sq_rel = sq_rel.mean()

        metrics_dict = {
            "abs_rel": abs_rel,
            "sq_rel": sq_rel,
            "rmse": rmse,
            "rmse_log": rmse_log,
            "a1": a1,
            "a2": a2,
            "a3": a3,
            "threshold": thresh.to("cpu"),
            "abs_err": torch.abs(depth_gt - depth_pred).to("cpu"),
            "rel_err": (torch.abs(depth_gt - depth_pred) / depth_gt).to("cpu"),
        }
        return metrics_dict


def evaluation(local_rank, config):
    return base_evaluation(local_rank, config, get_dataflow, initialize, get_metrics)


def get_dataflow(config):
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
    names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    num_bins = config.get("num_bins", sum(config.get("data")["image_size"]) // 2)
    metrics = {
        name: MeanMetric((lambda n: lambda x: x["output"][n])(name), device)
        for name in names
    }
    if config.get("hist_viz", False):  ## Error histogram
        metrics["abs_errH"] = HistogramMetric(
            "abs_err", num_bins, (lambda n: lambda x: x["output"][n])("abs_err")
        )
        metrics["rel_errH"] = HistogramMetric(
            "rel_err", num_bins, (lambda n: lambda x: x["output"][n])("rel_err")
        )
        metrics["thresholdH"] = HistogramMetric(
            "threshold", num_bins, (lambda n: lambda x: x["output"][n])("threshold")
        )
    return metrics


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "BTSNet")

    # Make configurable for Semantics as well
    sample_color = config["model_conf"].get("sample_color", True)
    d_out = 1 if sample_color else 4

    if arch == "MVBTSNet":
        if config["model_conf"]["code_mode"] == "z_feat":
            cam_pos = 1
        else:
            cam_pos = 0
        code_xyz = PositionalEncoding.from_conf(
            config["model_conf"]["code"], d_in=3 + cam_pos
        )
        encoder = make_backbone(config["model_conf"]["encoder"])
        d_in = (
            encoder.latent_size + code_xyz.d_out
        )  ### 103 | 116 (TODO: some issue in ids_encoding embedding in Tensor)
        decoder_heads = {
            head_conf["name"]: make_head(head_conf, d_in, d_out)
            for head_conf in config["model_conf"]["decoder_heads"]
        }

        if config["model_conf"]["decoder_heads"][0][
            "freeze"
        ]:  ## Freezing the MVhead for knowledge distillation
            for param in decoder_heads["multiviewhead"].parameters():
                param.requires_grad = False
            print("__frozen the MVhead for knowledge distillation.")
        else:
            print("__No freezing heads during training.")

        # net = globals()[arch]( config["model_conf"], ren_nc=config["renderer"]["n_coarse"], B_=config["batch_size"] )  ## default: globals()[arch](config["model_conf"])
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
        model = IBRNetModel(args, load_opt=False, load_scheduler=False)
        projector = Projector(device="cuda")
        net = IBRNetRenderingWrapper(model=model, projector=projector)

    elif arch == "neuray":
        pass

    else:
        net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()

    model = BTSWrapper(renderer, config["model_conf"])

    return model


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    pass
