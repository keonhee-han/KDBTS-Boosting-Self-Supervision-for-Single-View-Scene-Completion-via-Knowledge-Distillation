import math
from copy import copy
from typing import Optional, Union, Iterable, Sequence

import ignite.distributed as idist
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.dataloader import T_co, _collate_fn_t, _worker_init_fn_t
from torchvision.utils import make_grid

from datasets.data_util import make_datasets
from models.common.model.scheduler import make_scheduler
from models.common.render import NeRFRenderer
from models.bts.model.loss import ReconstructionLoss
from models.bts.trainer import get_metrics, BTSWrapper, MVBTSNet    ## default: BTSNet
from utils.array_operations import map_fn, unsqueezer, to
from utils.base_trainer import base_training
from utils.plotting import color_tensor

# from utils.hparam_tuning import random_search, grid_search
from models.bts.model.models_bts import MVBTSNet

"""
Approach: The transformer is also trained using the reconstruction loss from the BTS paper. The forward function of BTSNet and 
correspondingly our MultiViewBTSNet output (rbg, invalid, density) is then used in the renderer instance (NeRFRenderer) 
to synthesize a target Image. This target image is then compared to the "ground truth" image to calculate the reconstruction loss.

Replace the encoder: You currently have an EncoderDummy which always returns the same feature map. 
You would replace this with your Transformer model. The Transformer would take the sequence of views as input and return
a sequence of encoded views as output. The size of the output should match the expected size of the feature map in the 
next stage of your pipeline.
"""
## dummy encoder and a DataLoader that always returns the same batch of data
## to ensure that the feature representation is not a variable in your experiment.
# Since EncoderDummy always returns the same feature map, any learning or lack thereof
# can be attributed to the parts of the model after the encoder.

class EncoderDummy(nn.Module):  ## it returns a pre-defined feature tensor every time it is called.
    def __init__(self, size, feat_dim, num_views=1) -> None:
        super().__init__() ## initializes this feature map as a random tensor of a specified size
        self.feats = nn.Parameter(torch.randn(num_views, feat_dim, *size)) ## size:=determines the size of the feature map it produces
        # self.feats = nn.Parameter(torch.stack([torch.linspace(i, feat_dim - 1 + i, feat_dim)[:, None, None].repeat(1, *size) for i in range(num_views)], dim=0)) ## size:=determines the size of the feature map it produces
        self.latent_size = feat_dim

    def forward(self, x):   ## dim(x):= (B,C,H,W) == (1,64,192,640)     ### on thursday meeting to discuss  ## batch size should be 1
        n = x.shape[0]      ### torch.Size([4, 3, 192, 640])    4:= nv_, 3:=RGB
        # for b_in_img in self.feats:     ## dim(feats): (10,64,192,640)  ?? Note: batch, B, isn't the same as num of multiviews
        #     return [b_in_img.expand(n, -1, -1, -1)]
        return [self.feats.expand(n, -1, -1, -1)]   ## ? ## origin: repeat the fixed feature map n batch times along the first dimension,
        """
        Note: -1 for the other dimensions mean those dimensions are not expanded and will retain their original sizes.
        effectively creating a batch of identical feature maps. This makes it compatible with the rest of the model
        that might be expecting a batch of feature maps
        The forward method then takes an input x, ignores it, and returns the feature map repeated n times,
        where n is the first dimension of x (usually the batch size). This allows the EncoderDummy to be used
        in place of a real encoder in a model that expects to process batches of input data. ## all views will produce the same feature map
        """
class DataloaderDummy(DataLoader):  ## always returns the same element every time it's iterated over
    def __init__(self, dataset: Dataset[T_co], batch_size: Optional[int] = 1, shuffle: Optional[bool] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None, num_workers: int = 0,
                 collate_fn: Optional[_collate_fn_t] = None, pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None, multiprocessing_context=None,
                 generator=None, *, prefetch_factor: int = 2, persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)

        self.element = to(map_fn(map_fn(dataset.__getitem__(0), torch.tensor), unsqueezer), "cuda:0")

    def _get_iterator(self):
        return iter([self.element])

    def __iter__(self):
        return super().__iter__()

    def __len__(self) -> int:
        return 1


class BTSWrapperOverfit(BTSWrapper):
    def __init__(self, renderer, config, eval_nvs=False, size=None) -> None:
        super().__init__(renderer, config, eval_nvs)

        # self.encoder_dummy = EncoderDummy(size, config["encoder"]["d_out"], num_views=3)  ## origin
        self.encoder_dummy = EncoderDummy(size, config["encoder"]["d_out"], num_views=config["num_multiviews"])

        self.renderer.net.encoder = self.encoder_dummy
        self.renderer.net.flip_augmentation = False

## doesn't do any training itself, but it "sets" up the training process with a specified dataflow function(get_dataflow),
# initializer function (initialize), metrics function (get_metrics), and visualization function (visualize).
def training(local_rank, config):
    return base_training(local_rank, config, get_dataflow, initialize, get_metrics, visualize)


def get_dataflow(config, logger=None):
    # - Get train/test datasets
    if idist.get_local_rank() > 0:
        # Ensure that only local rank 0 download the dataset
        # Thus each node will download a copy of the datasetMVBTSNet
        idist.barrier()

    train_dataset, _ = make_datasets(config["data"])

    train_dataset.length = 1
    train_dataset._skip = config["data"].get("skip", 0)

    vis_dataset = copy(train_dataset)
    test_dataset = copy(train_dataset)
    ## it's not always needed when using the model to make predictions. Once the model has been trained, it can predict a 3D feature volume from a set of input images. These 3D features can then be used to synthesize new views of the scene, without needing any depth information.
    vis_dataset.return_depth = True        ## ! don't need for training but need it for evaluation (test) to quantify the loss metrics
    test_dataset.return_depth = True

    if idist.get_local_rank() == 0:
        # Ensure that only local rank 0 download the dataset
        idist.barrier()     ## Once the dataset has been downloaded, the barrier is invoked, and only then are the other processes allowed to proceed.
        ## By using this method, you can control the order of execution in a distributed setting and ensure that certain
        ## steps are not performed multiple times by different processes. This can be very useful when working with shared
        ## resources or when coordination is required between different processes.

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = DataloaderDummy(train_dataset)
    test_loader = DataloaderDummy(test_dataset)
    vis_loader = DataloaderDummy(vis_dataset)

    return train_loader, test_loader, vis_loader


def initialize(config: dict, logger=None):
    arch = config["model_conf"].get("arch", "MVBTSNet")         ## Model creation  ## origin: get("arch", "BTSNet")
    net = globals()[arch](config["model_conf"])
    renderer = NeRFRenderer.from_conf(config["renderer"], eval_batch_size=100000)       ## ! instance of a Neural Radiance Fields (NeRF) renderer using settings from the configuration
    renderer = renderer.bind_parallel(net, gpus=None).eval()    ## renderer is then bound to the model

    mode = config.get("mode", "depth")

    model = BTSWrapperOverfit(
        renderer,
        config["model_conf"],
        mode == "nvs",
        size=config["data"].get("image_size", (192, 640))
    )

    model = idist.auto_model(model) ## Distributed setup: The model is prepared for distributed training, tools for easy setup of distributed training

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = idist.auto_optim(optimizer)

    lr_scheduler = make_scheduler(config.get("scheduler", {}), optimizer)

    criterion = ReconstructionLoss(config["loss"], config["model_conf"].get("use_automasking", False))

    return model, optimizer, criterion, lr_scheduler


def visualize(engine: Engine, logger: TensorboardLogger, step: int, tag: str):
    print("Visualizing")

    data = engine.state.output["output"]
    writer = logger.writer

    images = torch.stack(data["imgs"], dim=1).detach()[0]
    recon_imgs = data["fine"][0]["rgb"].detach()[0]
    recon_depths = [f["depth"].detach()[0] for f in data["fine"]]

    depth_profile = data["coarse"][0]["alphas"].detach()[0]
    alphas = data["coarse"][0]["alphas"].detach()[0]
    invalids = data["coarse"][0]["invalid"].detach()[0]

    z_near = data["z_near"]
    z_far = data["z_far"]

    take = list(range(0, images.shape[0], 2))

    _, c, h, w = images.shape
    nv = recon_imgs.shape[0]

    images = images[take]
    images = images * .5 + .5

    recon_imgs = recon_imgs.view(nv, h, w, -1, c)
    recon_imgs = recon_imgs[take]
    # Aggregate recon_imgs by taking the mean
    recon_imgs = recon_imgs.mean(dim=-2).permute(0, 3, 1, 2)

    recon_mse = (((images - recon_imgs) ** 2) / 2).mean(dim=1).clamp(0, 1)
    recon_mse = color_tensor(recon_mse, cmap="plasma").permute(0, 3, 1, 2)

    recon_depths = [(1 / d[take] - 1 / z_far) / (1 / z_near - 1 / z_far) for d in recon_depths]
    recon_depths = [color_tensor(d.squeeze(1).clamp(0, 1), cmap="plasma").permute(0, 3, 1, 2) for d in recon_depths]

    depth_profile = depth_profile[take][:, [h//4, h//2, 3*h//4], :, :].view(len(take)*3, w, -1).permute(0, 2, 1)
    depth_profile = depth_profile.clamp_min(0) / depth_profile.max()
    depth_profile = color_tensor(depth_profile, cmap="plasma").permute(0, 3, 1, 2)

    alphas = alphas[take]

    alphas += 1e-5

    ray_density = alphas / alphas.sum(dim=-1, keepdim=True)
    ray_entropy = -(ray_density * torch.log(ray_density)).sum(-1) / (math.log2(alphas.shape[-1]))
    ray_entropy = color_tensor(ray_entropy, cmap="plasma").permute(0, 3, 1, 2)

    alpha_sum = (alphas.sum(dim=-1) / alphas.shape[-1]).clamp(-1)
    alpha_sum = color_tensor(alpha_sum, cmap="plasma").permute(0, 3, 1, 2)

    invalids = invalids[take]
    invalids = invalids.mean(-2).mean(-1)
    invalids = color_tensor(invalids, cmap="plasma").permute(0, 3, 1, 2)

    # Write images
    nrow = int(len(take) ** .5)

    images_grid = make_grid(images, nrow=nrow)
    recon_imgs_grid = make_grid(recon_imgs, nrow=nrow)
    recon_depths_grid = [make_grid(d, nrow=nrow) for d in recon_depths]
    depth_profile_grid = make_grid(depth_profile, nrow=nrow)
    ray_entropy_grid = make_grid(ray_entropy, nrow=nrow)
    alpha_sum_grid = make_grid(alpha_sum, nrow=nrow)
    recon_mse_grid = make_grid(recon_mse, nrow=nrow)
    invalids_grid = make_grid(invalids, nrow=nrow)

    writer.add_image(f"{tag}/input_im", images_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_im", recon_imgs_grid.cpu(), global_step=step)
    for i, d in enumerate(recon_depths_grid):
        writer.add_image(f"{tag}/recon_depth_{i}", d.cpu(), global_step=step)
    writer.add_image(f"{tag}/depth_profile", depth_profile_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/ray_entropy", ray_entropy_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/alpha_sum", alpha_sum_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/recon_mse", recon_mse_grid.cpu(), global_step=step)
    writer.add_image(f"{tag}/invalids", invalids_grid.cpu(), global_step=step)
