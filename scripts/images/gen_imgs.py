import sys

sys.path.append(".")  ## to make system recognize the module name to be imported
from models.common.backbones.backbone_util import make_backbone
from models.common.model.code import PositionalEncoding
from models.common.model.head_util import make_head
from models.ibrnet.config import config_parser
from models.ibrnet.ibrwrapper import IBRNetRenderingWrapper
from models.ibrnet.model import IBRNetModel
from models.ibrnet.projection import Projector


from scripts.inference_setup import *

import copy

import hydra
import torch

from models.bts.model import MVBTSNet, ImageRaySampler
from models.common.render import NeRFRenderer
from utils.array_operations import map_fn, unsqueezer
from utils.plotting import color_tensor


def main():
    s_img = True
    s_depth = True     ## default: True
    s_profile = True   ## default: True
    dry_run = False

    task = "KITTI-Raw"
    # task = "KITTI-360"
    dataset_split = "test"   ## test | train
    model_type = "DFT"  ## default: MVBTS | KDBTS | DFT | IBR
    enc_type = "stereo"  ## mono | mono_temporal | stereo | stereo_temporal | stereo_fisheye | stereo_fisheye_alltemporal
    indices_type = "custom"    ## all | custom
    
    kwargs = {
        "split": dataset_split,
        "model_type": model_type,
        "enc_type": enc_type,
        "indices_type": indices_type,
    }

    assert task in [
        "KITTI-360",
        "KITTI-Raw",
        "RealEstate10K",
    ]  ## dataset, not model type

    if task == "KITTI-360":
        (
            dataset,
            config_path,
            cp_path,
            out_path,
            resolution,
            cam_incl_adjust,
        ) = setup_kitti360("imgs", **kwargs)
    elif task == "KITTI-Raw":
        (
            dataset,
            config_path,
            cp_path,
            out_path,
            resolution,
            cam_incl_adjust,
        ) = setup_kittiraw("imgs", **kwargs)
    elif task == "RealEstate10K":
        (
            dataset,
            config_path,
            cp_path,
            out_path,
            resolution,
            cam_incl_adjust,
        ) = setup_re10k("imgs")
    else:
        raise ValueError(f"Invalid task: {task}")

    if indices_type == "all":
        indices = range(len(dataset))  ## retrieving all imgs in the dataset
    elif indices_type == "custom":
        # indices = [1 * i for i in range(0, 670)]
        indices = [30, 51, 70, 102, 121, 214, 294, 361, 418, 475]   ## KITTI-Raw for Fig. 11 in BTS paper
        # indices = [16, 40, 54, 290, 367, 435, 539, 591, 627, 651, 654]        ## KITTI-Raw
        # indices = [110, 190]
        print(f"__viz desired indices:{indices}")
    else: NotImplementedError(f"Unsupported input: {indices_type}")

    # Slightly hacky, but we need to load the config based on the task
    global config
    config = {}

    print(
        f"_dataset_task:  {task}\n_model_type:  {model_type}\n_encoding_type:  {enc_type}\n_indices_type:  {indices_type}\n_dataset_split:  {dataset_split}\n_show_img:  {s_img}\n_show_depth:  {s_depth}\n_show_profile:  {s_profile}\n_dry_run:  {dry_run}"
    )

    @hydra.main(version_base=None, config_path="../../configs", config_name=config_path)
    def main_dummy(cfg):
        global config
        config = copy.deepcopy(cfg)

    main_dummy()  ## ?

    print("__Setup folders")
    out_path.mkdir(exist_ok=True, parents=True)

    print("__Loading checkpoint")
    cp = torch.load(cp_path, map_location=device)

    arch = config["model_conf"].get("arch", "BTSNet")
    sample_color = config["model_conf"].get("sample_color", True)
    d_out = 1 if sample_color else 4

    # enc_type = config["model_conf"].get("arch", "enc_type")
    if enc_type == "mono":
        enc_in_ids = [0]
    elif enc_type == "mono_temporal":
        enc_in_ids = [0, 1]
    elif enc_type == "stereo":
        enc_in_ids = [0, 2]
    elif enc_type == "stereo_temporal":
        enc_in_ids = [0, 1, 2, 3]
    elif enc_type == "stereo_fisheye" and task == "KITTI-360":
        enc_in_ids = [0, 2, 4, 6]  ## ! check fisheye indices
    elif enc_type == "stereo_fisheye_alltemporal" and task == "KITTI-360":
        enc_in_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        raise NotImplementedError(
            f"Unrsupported encoder type:{enc_type} for dataset task{task}"
        )

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
        model = IBRNetModel(args, load_opt=False, load_scheduler=False)
        projector = Projector(device="cuda")
        net = IBRNetRenderingWrapper(model=model, projector=projector)

    elif arch == "neuray":
        pass

    else:
        net = globals()[arch](config["model_conf"])

    renderer = NeRFRenderer.from_conf(config["renderer"])
    renderer = renderer.bind_parallel(net, gpus=None).eval()
    renderer.renderer.n_coarse = 64
    renderer.renderer.lindisp = True

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.renderer = renderer

    _wrapper = _Wrapper()

    if arch != "IBRNet":
        _wrapper.load_state_dict(cp["model"], strict=False)
    renderer.to(device)
    renderer.eval()

    ray_sampler = ImageRaySampler(
        config["model_conf"]["z_near"],
        config["model_conf"]["z_far"],
        *resolution,
        norm_dir=False,
    )

    with torch.no_grad():
        for idx in indices:
            data = dataset[idx]
            data_batch = map_fn(map_fn(data, torch.tensor), unsqueezer)
            images = torch.stack(data_batch["imgs"], dim=1).to(device)
            poses = torch.stack(data_batch["poses"], dim=1).to(device)
            projs = torch.stack(data_batch["projs"], dim=1).to(device)
            img = images[0, 0].permute(1, 2, 0).cpu() * 0.5 + 0.5
            load_from_file = False

            if isinstance(renderer.net, IBRNetRenderingWrapper):
                n_samples = 64
                renderer.net.model.net_coarse.pos_encoding = (
                    renderer.net.model.net_coarse.posenc(d_hid=16, n_samples=n_samples)
                )
                renderer.net.model.args.N_samples = n_samples
                renderer.net.regular_grid = False

            if load_from_file:
                ibrnet = True
                if ibrnet:
                    base_folder = Path(
                        "/storage/user/hank/methods_test/IBRNet/kittiraw/savdpred/kittiraw_250000"
                    )
                    depth = torch.load(base_folder.joinpath(f"{idx}_depth_gray.pt")).to(
                        device
                    )[None]
                else:
                    base_folder = Path(
                        "/storage/user/hank/methods_test/DevNet/test_imgs"
                    )  ##
                    depth = torch.from_numpy(
                        torch.load(base_folder.joinpath(f"Tensor{idx}_kitti_raw.pt"))
                    ).to(device)
                normalize_gt = True
                calibration_offset = True
                if normalize_gt:  ## depth scailing for viz
                    gt_depth = torch.stack(data_batch["depths"], dim=1).to(device)
                    depth_scaled = torch.nn.functional.interpolate(
                        depth[None], tuple(gt_depth.shape[-2:])
                    )
                    mask = gt_depth > 0
                    if calibration_offset:
                        scaling = torch.median(gt_depth[mask]) / torch.median(
                            depth_scaled[None][mask]
                        )
                        depth = (scaling * depth)[0].to("cpu")
            else:
                # Move coordinate system to input frame
                poses = torch.inverse(poses[:, :1, :, :]) @ poses

                net.encode(images, projs, poses, ids_encoder=enc_in_ids, ids_render=[0])
                net.set_scale(0)

                _, depth = render_poses(
                    renderer, ray_sampler, poses[:, :1], projs[:, :1]
                )

            if s_profile:
                profile = render_profile(net, cam_incl_adjust)
            else:
                profile = None

            depth = (
                (1 / depth - 1 / config["model_conf"]["z_far"])
                / (
                    1 / config["model_conf"]["z_near"]
                    - 1 / config["model_conf"]["z_far"]
                )
            ).clamp(0, 1)

            print(f"Generated " + str(out_path / f"{idx:010d}"))

            if s_img:
                save_plot(
                    img.numpy(), str(out_path / f"{idx:010d}_in.png"), dry_run=dry_run
                )
            if s_depth:
                save_plot(
                    color_tensor(depth, "magma", norm=True).numpy(),
                    str(out_path / f"{idx:010d}_depth.png"),
                    dry_run=dry_run,
                )
            if s_profile:
                save_plot(
                    color_tensor(profile.cpu(), "magma", norm=True).numpy(),
                    str(out_path / f"{idx:010d}_profile.png"),
                    dry_run=dry_run,
                )


if __name__ == "__main__":
    main()
