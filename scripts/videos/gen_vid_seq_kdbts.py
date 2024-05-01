import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm

import sys
from argparse import ArgumentParser
from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.head_util import make_head

sys.path.append(".")

from scripts.inference_setup import *

import copy

import hydra
import torch

from models.bts.model import BTSNet, MVBTSNet, ImageRaySampler
from models.common.render import NeRFRenderer
from utils.array_operations import map_fn, unsqueezer
from utils.plotting import color_tensor


def main():
    # parser = ArgumentParser("Specify configs for video generation.")
    # parser.add_argument("--task", "-t", help="task to use dataset. (KITTI-360 (default), KITTI-Raw", default="KITTI-360")
    # parser.add_argument("--model", "-m", help="Which pretrained model to use. (kdbts (default), mvbts", default="kdbts")
    # args = parser.parse_args()

    s_img = True
    s_depth = True
    s_profile = True
    dry_run = False

    # task, model = args.task, args.model
    # task, model = "KITTI-360", "kdbts"
    # task, model = "KITTI-360", "mvbts"
    task, model = "KITTI-360", "bts"
    # task, model = "KITTI-360", "pixelnerf"

    # task, model = "KITTI-Raw", "kdbts"
    # task, model = "KITTI-Raw", "mvbts"
    # task, model = "KITTI-Raw", "bts"
    assert task in ["KITTI-360", "KITTI-Raw"]
    assert model in ["kdbts", "mvbts", "bts", "pixelnerf"]

    seq_ =  "2013_05_28_drive_0000_sync" 
    # seq_ =  "2013_05_28_drive_0002_sync"    ## 0-1854

    # seq_ =  "2013_05_28_drive_0003_sync" 
    # seq_ =  "2013_05_28_drive_0005_sync" 
    # seq_ =  "2013_05_28_drive_0007_sync" 
    # seq_ =  "2013_05_28_drive_0010_sync"
    # seq_ =  "2013_05_28_drive_0004_sync" 
    # seq_ =  "2013_05_28_drive_0006_sync" 
    # seq_ =  "2013_05_28_drive_0009_sync"

    if seq_ == "2013_05_28_drive_0002_sync":
         FROM, TO = 0, 1854
    elif seq_ == "2013_05_28_drive_0000_sync": 
         FROM, TO = 0, 2400
        #  FROM, TO = 0, 50
    # FROM = 2000
    # # TO = 1400
    # # TO = 2100
    # TO = 2400
    assert 0 <= FROM < TO

    d_min, d_max = 3, 30        ## Note: increasing the range of scene rendering will increase the computation. default in BTS paper setting: 23

    if task == "KITTI-360":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = (
            setup_kitti360("videos/seq", seq_, "val_seq", model)
        )
    elif task == "KITTI-Raw":
        dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = (
            setup_kittiraw("videos/seq", "test")
        )
    # elif task == "RealEstate10K":
    #     dataset, config_path, cp_path, out_path, resolution, cam_incl_adjust = setup_re10k("videos/seq", "test")
    else:
        raise ValueError(f"Invalid task: {task}")

    # Slightly hacky, but we need to load the config based on the task
    global config
    config = {}

    @hydra.main(version_base=None, config_path="../../configs", config_name=config_path)
    def main_dummy(cfg):
        global config
        config = copy.deepcopy(cfg)

    main_dummy()

    print("Setup folders")
    out_path.mkdir(exist_ok=True, parents=True)
    # file_name = f"{FROM:05d}-{TO:05d}.mp4"
    file_name = f"{FROM:05d}-{TO:05d}_{task}_{model}_{seq_}_{d_min}_{d_max}.mp4"      ##

    print("Loading checkpoint")
    cp = torch.load(cp_path, map_location=device)

    if config["model_conf"]["arch"] == "MVBTSNet":
        d_out = 1
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

        net = MVBTSNet(
            config["model_conf"],
            encoder,
            code_xyz,
            decoder_heads,
            # final_pred_head=config.get("final_prediction_head", None),
            final_pred_head=config["model_conf"].get("final_prediction_head", None),
            ren_nc=config["renderer"]["n_coarse"],
        )

    elif config["model_conf"]["arch"] == "BTSNet":    ## For single view BTS model
        net = BTSNet(config["model_conf"])
        encoder = make_backbone(config["model_conf"]["encoder"])
        code_xyz = PositionalEncoding.from_conf(config["model_conf"]["code"], d_in=3)
        d_in = encoder.latent_size + code_xyz.d_out  ### 103

    else:  
        raise ValueError(f"__Invalid architecture: {config['model_conf']['arch']}")
    
    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    _wrapper.load_state_dict(cp["model"], strict=False)
    renderer.to(device)
    renderer.eval()

    ray_sampler = ImageRaySampler(
        config["model_conf"]["z_near"],
        config["model_conf"]["z_far"],
        *resolution,
        norm_dir=False,
    )

    # Change resolution to match final height
    if s_depth and s_img:
        OUT_RES.P_RES_ZX = (resolution[0] * 2, resolution[0] * 2)
    else:
        OUT_RES.P_RES_ZX = (resolution[0], resolution[0])

    frames = []

    with torch.no_grad():
        for idx in tqdm(range(FROM, TO, 2)):
            data = dataset[idx]
            data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)

            images = torch.stack(data_batch["imgs"], dim=1).to(device)
            poses = torch.stack(data_batch["poses"], dim=1).to(device)
            projs = torch.stack(data_batch["projs"], dim=1).to(device)

            # Move coordinate system to input frame
            poses = torch.inverse(poses[:, :1, :, :]) @ poses

            # net.encode(images, projs, poses, ids_encoder=[0, 1], ids_render=[0])
            ren_views = config["model_conf"]["ids_enc_offset_viz"]
            net.encode(images, projs, poses, ids_encoder=ren_views, ids_render=[0], model_name = model)   
            # net.encode(images, projs, poses, ids_encoder=ren_views, ids_render=ren_views)  
            net.set_scale(0)

            img = images[0, 0].permute(1, 2, 0).cpu() * 0.5 + 0.5

            img = img.numpy()

            if s_depth:
                _, depth = render_poses(
                    renderer, ray_sampler, poses[:, :1], projs[:, :1]
                )
                depth = 1 / depth
                depth = ((depth - 1 / d_max) / (1 / d_min - 1 / d_max)).clamp(0, 1)
                depth = color_tensor(depth, "magma", norm=False).numpy()
            else:
                depth = None

            if s_profile:
                profile = render_profile(net, cam_incl_adjust, d_min=d_min, d_max=d_max)
                profile = color_tensor(profile.cpu(), "magma", norm=True).numpy()
            else:
                profile = None

            if s_img and s_depth and s_profile:
                frame = np.concatenate((img, depth), axis=0)
                frame = np.concatenate((frame, profile), axis=1)
            elif s_img and s_depth:
                frame = np.concatenate((img, depth), axis=0)
            elif s_img and s_profile:
                frame = np.concatenate((img, profile), axis=1)
            elif s_img:
                frame = img
            elif s_depth:
                frame = depth
            elif s_profile:
                frame = profile
            else:
                frame = None
            frames.append(frame)

    frames = [(frame * 255).astype(np.uint8) for frame in frames]

    if not dry_run:
        video = ImageSequenceClip(frames, fps=10)
        video.write_videofile(str(out_path / file_name))
        video.close()

    print("Completed.")


if __name__ == "__main__":
    main()