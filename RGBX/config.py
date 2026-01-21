import os
import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict
import argparse
from pathlib import Path


C = edict()
config = C
cfg = C

# Random seed
C.seed = 12345

# Repository root: compute from this file's location so config is independent of CWD
REPO_ROOT = Path(__file__).resolve().parents[1]
C.root_dir = str(REPO_ROOT)
# Absolute dir (kept for backward compatibility)
C.abs_dir = osp.realpath(str(REPO_ROOT))

# Dataset config
C.dataset_name = "HillfortDataSet"
C.dataset_path = osp.join(C.root_dir, "datasets", C.dataset_name)
# RGB imagery (tiles) — adjust format if your tiles are .tif or .jpg
# Paths are absolute and resolved from the repository root
C.rgb_root_folder = osp.join(C.dataset_path, "RGB")
C.rgb_format = ".png"
# Ground-truth masks (single-channel PNGs)
C.gt_root_folder = osp.join(C.dataset_path, "Label")
C.gt_format = ".png"
C.gt_transform = False
# Auxiliary input (DTM) — single-channel GeoTIFFs
C.x_root_folder = osp.join(C.dataset_path, "DTM")
C.x_format = ".tif"
C.x_is_single_channel = True  # DTM is single-channel
C.train_source = osp.join(C.dataset_path, "train.txt")
C.eval_source = osp.join(C.dataset_path, "test.txt")
C.is_test = False
# Auto-detect counts from train/test files when available
C.num_train_imgs = 0
C.num_eval_imgs = 0
try:
    if osp.exists(C.train_source):
        with open(C.train_source, "r") as f:
            C.num_train_imgs = sum(1 for l in f if l.strip())
    if osp.exists(C.eval_source):
        with open(C.eval_source, "r") as f:
            C.num_eval_imgs = sum(1 for l in f if l.strip())
except Exception:
    C.num_train_imgs = 0
    C.num_eval_imgs = 0

# Binary segmentation: background + hillfort
C.num_classes = 2
C.class_names = ["background", "hillfort"]

"""Image Config"""
C.background = 0
C.image_height = 512
C.image_width = 512
C.norm_mean = np.array([0.485, 0.456, 0.406])
C.norm_std = np.array([0.229, 0.224, 0.225])

""" Settings for network, this would be different for each kind of model"""
C.backbone = "mit_b2"
C.pretrained_model = C.root_dir + "/models/segformers/mit_b2.pth"
C.decoder = "MLPDecoder"
C.decoder_embed_dim = 512
C.optimizer = "AdamW"

"""Train Config"""
C.lr = 6e-5
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 0.01
C.batch_size = 8  # 8
C.nepochs = 40  # 50
C.niters_per_epoch = C.num_train_imgs // C.batch_size + 1
C.num_workers = 0  # 4
C.train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
C.warm_up_epoch = 10

# Stratified sampling settings (optional)
# path to folder containing neg.txt, small_pos.txt, mid_pos.txt, full_pos.txt
C.stratified_buckets_dir = C.root_dir + "/datasets/" + C.dataset_name
# Adjust sampling proportions to upweight positive samples (neg, small, mid, full)
# Default was [0.5, 0.4, 0.09, 0.01] which led to very few positive tiles in batches.
# New proportions increase the share of positive buckets so training sees more positives.
C.stratified_proportions = [0.5, 0.25, 0.15, 0.10]
# Toggle to indicate we explicitly want to oversample positives; training/sampler
# code can read this flag to apply oversampling or duplication of positive entries.
C.oversample_positives = True
# Optional: factor by which to oversample positive buckets (1=no change)
C.positive_oversample_factor = 3
# Use pixel-level weights computed from `tile_stats.csv` when available.
# If True, training computes class weights from per-tile positive counts
# for the training split. Set to False to fall back to tile-count-based weighting.
C.use_pixel_weights = True
# Class-weight clipping to avoid extreme weighting from very sparse positives
# Maximum allowed per-class weight after normalization (helps stability)
C.max_class_weight = 50.0
# Minimum allowed per-class weight after normalization (prevents background from vanishing)
C.min_class_weight = 0.01
# Loss settings
C.loss_type = "dice_ce"
C.dice_weight = 1.0
C.ce_weight = 1.0

# Per-batch downweighting for very-full tiles
C.full_pos_frac_threshold = 0.9
C.full_pos_downweight = 0.5

C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1

"""Eval Config"""
C.eval_iter = 25
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1]  # [0.75, 1, 1.25] #
C.eval_flip = False  # True #
C.eval_crop_size = [512, 512]  # [height weight]

"""Store Config"""
C.checkpoint_start_epoch = 10
C.checkpoint_step = 5

"""Path Config"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.join(C.root_dir))

# Place logs inside the project root to avoid depending on CWD
C.log_dir = osp.abspath(
    osp.join(C.root_dir, "log_" + C.dataset_name + "_" + C.backbone)
)
C.tb_dir = osp.abspath(osp.join(C.log_dir, "tb"))
C.log_dir_link = C.log_dir
C.checkpoint_dir = osp.abspath(osp.join(C.log_dir, "checkpoint"))

exp_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.log_file = C.log_dir + "/log_" + exp_time + ".log"
C.link_log_file = C.log_file + "/log_last.log"
C.val_log_file = C.log_dir + "/val_" + exp_time + ".log"
C.link_val_log_file = C.log_dir + "/val_last.log"

if __name__ == "__main__":
    print(config.nepochs)
    parser = argparse.ArgumentParser()
    parser.add_argument("-tb", "--tensorboard", default=False, action="store_true")
    args = parser.parse_args()

    if args.tensorboard:
        open_tensorboard()
