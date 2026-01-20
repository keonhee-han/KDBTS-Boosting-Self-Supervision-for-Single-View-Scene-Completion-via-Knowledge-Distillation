# KDBTS: Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation

[**Paper**](https://arxiv.org/abs/2404.07933) |  [**Video**](https://youtu.be/iWJmIwjtWfI?si=Lq3QSLfw-NGNnRa4) | [**Project Page**](https://keonhee-han.github.io/publications/kdbts/)
This is the official implementation for the CVPR 2024 paper:

> **KDBTS: Boosting Self-Supervision for Single-View Scene Completion via Knowledge Distillation**
>
> [Keonhee Han](https://keonhee-han.github.io)<sup>1</sup>, [Dominik Muhle](https://dominikmuhle.github.io)<sup>1,2</sup>, [Felix Wimbauer](https://fwmb.github.io)<sup>1,2</sup>, and [Daniel Cremers](https://vision.in.tum.de/members/cremers)<sup>1,2</sup><br>
> <sup>1</sup>Technical University of Munich, <sup>2</sup>Munich Center for Machine Learning 
> 
> [**CVPR 2024** (arXiv)](https://arxiv.org/abs/2404.07933)

If you find our work useful, please consider citing our paper:
```
@inproceedings{han2024kdbts,
  title = {Boosting Self-Supervision for Single-View Scene Completion
  via Knowledge Distillation},
  author = {K Han and D Muhle and F Wimbauer and D Cremers},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024},
  eprint = {2404.07933},
  eprinttype = {arXiv},
  eprintclass = {cs.CV},
 }
```

Occupancy Preidction in KITTI with baseline comparison

<!--
<video width="100%" autoplay muted loop>
  <source src="https://github.com/keonhee-han/keonhee-han.github.io/blob/main/_publications/kdbts/assets/kdbts_demo_small_size.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
-->

[![Baseline_comparison](https://github.com/keonhee-han/keonhee-han.github.io/blob/main/_publications/kdbts/assets/thumbnail_kdbts_maxresdefault.png)](https://youtu.be/vNtLwpp_LlA?si=ghPzzDMIS5QnzphA)

# 📋 Abstract 
Inferring scene geometry from images via Structure from Motion is a long-standing and fundamental problem in computer vision. While classical approaches and, more recently, depth map predictions only focus on the visible parts of a scene, the task of scene completion aims to reason about geometry even in occluded regions. With the popularity of <cite>[neural radiance fields (NeRFs)][1]</cite>, implicit representations also became popular for scene completion by predicting so-called density fields. Unlike explicit approaches e.g. voxel-based methods, density fields also allow for accurate depth prediction and novel-view synthesis via image based rendering. In this work, we propose to fuse the scene reconstruction from multiple images and distill this knowledge into a more accurate single-view scene reconstruction. To this end, we propose Multi-View Behind the Scenes(MVBTS) to fuse density fields from multiple posed images, trained fully self-supervised only from image data. Using knowledge distillation, we use MVBTS to train a single view scene completion network via direct supervision called KDBTS. It achieves state-of-the-art performance on occupancy prediction, especially in occluded regions.

![Overview](https://github.com/keonhee-han/keonhee-han.github.io/blob/main/_publications/kdbts/assets/overview.png?raw=true)

***Figure 1. Overview** Knowledge Distillation from Multi-View to Single-View.* 

We propose to boost single-view scene completion by exploiting additional information from multiple images. 

**a)** we first train a novel multi-view scene reconstruction algorithm that is able to fuse density fields from multiple images in a fully self-supervised manner. 

**b)** we then employ knowledge distillation to directly supervise a state-of-the-art single-view reconstruction model in 3D to boost its performance.

# 🏗️️ Setup

### 🐍 Python Environment

We use **Conda** to manage our Python environment:
```shell
conda env create -f environment.yml
```
Then, activate the conda environment :
```shell
conda activate kdbts
```

### 💾 Datasets

All data should be placed under the `data/` folder (or linked to there) in order to match our config files for the 
different datasets.
The folder structure should look like:

```bash
data/KITTI-360
data/KITTI-Raw
```

All non-standard data (like precomputed poses and datasplits) comes with this repository and can be found in the `datasets/` folder.

**KITTI-360**

To download KITTI-360, go to https://www.cvlibs.net/datasets/kitti-360/index.php and create an account.
We require the perspective images, fisheye images, raw velodyne scans, calibrations, and vehicle poses.

**KITTI (Raw)**

To download KITTI, go to https://www.cvlibs.net/datasets/kitti/raw_data.php and create an account.
We require all synched+rectified data, as well as the calibrations.
The website also provides scripts for automatic downloading of the different sequences.
As we have found the provided ground truth poses to be lacking in quality, we computed our own poses with ORB-SLAM3 and use them by default.
They can be found under `datasets/kitti_raw/orb-slam_poses`.

**Other Dataset Implementations**

This repository contains dataloader implementations for other datasets, too. 
These are **not officially supported** and are **not guaranteed to work out of the box**.
However, they might be helpful when extending this codebase.

### 📸 Checkpoints

We provide download links for pretrained models for **KITTI-360**, **KITTI**.
Models will be stored under `out/<dataset>/pretrained/<checkpoint-name>.pth`.

```shell
download_checkpoint.sh {kitti-360|kitti-raw}
```

# 🏃 Running the Example

We provide a script to run our pretrained models with custom data.
The script can be found under `scripts/images/gen_img_custom.py` and takes the following flags:

- `--img <path>` / `i <path>`: Path to input image. The image will be resized to match the model's default resolution.
- `--model <model>` / `-m  <model>`: Which pretrained model to use (`KITTI-360` (default), `KITTI-Raw`).
- `--plot` / `-p`: Plot outputs instead of saving them.

`media/example/` contains two example images. Note that we use the default projection matrices for the respective datasets 
to compute the density profiles (birds-eye views). 
Therefore, if your custom data comes from a camera with different intrinsics, the output profiles might be skewed.

```bash
# Plot outputs
python scripts/images/gen_img_custom.py --img media/example/0000.png --model KITTI-360 --plot

# Save outputs to disk
python scripts/images/gen_img_custom.py --img media/example/0000.png --model KITTI-360
```

# 🏋 Training

We provide training configurations for our different models. For further detail about the configuration, please refer to: configs/readme.md
Generally, all trainings are run on a single Nvidia A40 GPU with 48GB memory.

**KITTI-360**

```bash
python train.py -cn train_kitti_360_base
```
For Knowledge distillation:
```bash
python train.py -cn train_kitti_360_KD
```

**KITTI (Raw)**

```bash
python train.py -cn train_kitti_raw_base
```
For Knowledge distillation:
```bash
python train.py -cn train_kitti_raw_KD
```

<!-- **RealEstate10K**

```bash
python train.py -cn exp_re10k
``` -->

# 📊 Evaluation

We further provide configurations to reproduce the evaluation results from the paper for occupancy and depth estimation.

```bash
# KITTI-360 Lidar Occupancy
python eval.py -cn eval_lidar_occ

# KITTI Raw Depth
python eval.py -cn eval_depth
```

# 📽 Rendering Images & Videos

We provide scripts to generate images and videos from the outputs of our models.
Generally, you can adapt the model and configuration for the output by changing some constant in the scripts.
Generated files are stored under `media/`.

**Inference on custom images**

Please refer to the example section.

**Generate images for samples from the datasets**
```bash
python scripts/images/gen_imgs.py
```
**Generate depth / profile videos**
```bash
python scripts/videos/gen_vid_seq_kdbts.py
```
**Generate novel view animations**
```bash
python scripts/videos/gen_vid_nvs.py
```
We provide different camera trajectories under `scripts/videos/trajectories`.

**Generate animation from depth map to top-down profile**
```bash
python scripts/videos/gen_vid_transition.py
```

# 🗣️ Acknowledgements

This work was supported by the ERC Advanced Grant SIMULACRON, by the Munich Cen- ter for Machine Learning, and by the German Federal Min- istry of Transport and Digital Infrastructure (BMDV) under grant 19F2251F for the ADAM project.

This repository is based on the [Behind The Scenes](https://github.com/Brummi/BehindTheScenes) code base and have inspirations from [IBRNet](https://github.com/googleinterns/IBRNet), [GeoNeRF](https://github.com/idiap/GeoNeRF), and [PixelNeRF](https://github.com/sxyu/pixel-nerf).

<!-- # TODO
1. Code brush up official code base for reproducibility.
2. Uploading pretrained KDBTS model

## Maybe Extra:
1. Make efficient transformer for faster training c.f. EfficientViT
To choose one of GPUs in the local machine, set the following environment variables:
```bash
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
``` -->
