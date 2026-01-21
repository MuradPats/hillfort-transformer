import os
from easydict import EasyDict as edict

# Repro
seed = 123

# Paths (adjust as needed)
# If you used these in RGBX/config.py, keep the same names to reuse dataloader behavior.
dataset_path = os.getenv("DATASET_PATH", "data/working/tiles")
root_dir = os.getenv("ROOT_DIR", "data/working")  # if your dataset code expects it

# Training
#nepochs = 50
#niters_per_epoch = 500
#checkpoint_start_epoch = 1
#checkpoint_step = 5
lr = 6e-4
lr_power = 0.9
warm_up_epoch = 1
nepochs = 1
niters_per_epoch = 5
checkpoint_start_epoch = 1
checkpoint_step = 1

optimizer = "AdamW"   # "SGDM" also supported
weight_decay = 0.01
momentum = 0.9        # only used by SGDM

# Segmentation
num_classes = 2  # background=0, hillfort=1
image_height = 512
image_width = 512

# Loss
loss_type = "dice_ce"
dice_weight = 1.0
ce_weight = 1.0

# Ignore index sentinel
ignore_index = 255

# Optional class/pixel weighting (keep same knobs as your RGBX train loop expects)
oversample_positives = False
positive_oversample_factor = 1.0
use_pixel_weights = False
max_class_weight = 100.0
min_class_weight = 0.0

# Per-batch downweight for overwhelmingly positive tiles (keep same behavior)
full_pos_frac_threshold = 0.90
full_pos_downweight = 0.25

# Baseline model options
baseline_use_modal_x = False   # True -> concat RGB + modal_x
modal_x_channels = 1           # MUST match minibatch["modal_x"] channels
baseline_base_channels = 64
baseline_bilinear = True

# Logging / checkpoints (reuse same conventions)
tb_dir = "runs/baseline_unet"
checkpoint_dir = "runs/baseline_unet/checkpoints"
log_dir = "runs/baseline_unet/logs"
log_dir_link = "runs/baseline_unet/log_latest"


# If this file already defines lots of variables like lr=..., nepochs=..., etc,
# this will wrap them into a single `config` object.
config = edict({k: v for k, v in globals().items()
               if not k.startswith("_") and k not in ("edict", "config")})
