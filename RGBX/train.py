import os.path as osp
import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor

from tensorboardX import SummaryWriter
from utils.loss_opr import DiceCrossEntropyLoss

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ["MASTER_PORT"] = "169710"

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + "/{}".format(
            time.strftime("%b%d_%d-%H-%M", time.localtime())
        )
        generate_tb_dir = config.tb_dir + "/tb"
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    # Use 255 as ignore_index (safe sentinel not present in remapped 0/1 masks)
    if getattr(config, "loss_type", "dice_ce") == "dice_ce":
        # Compute class weights when oversampling positives is enabled.
        # Prefer deriving weights from stratified bucket counts (neg vs positive)
        # if `stratified_buckets_dir` exists; otherwise fall back to
        # `positive_oversample_factor` for a simple upweight.
        weight = None
        try:
            # If enabled, try computing pixel-level weights from a tile_stats.csv file.
            if getattr(config, "use_pixel_weights", False):
                try:
                    # candidate locations for tile_stats.csv
                    candidates = [
                        Path(getattr(config, "dataset_path", "")) / "tile_stats.csv",
                        Path(getattr(config, "root_dir", "")) / "tile_stats.csv",
                    ]
                    csv_path = None
                    for c in candidates:
                        if c and c.exists():
                            csv_path = c
                            break

                    if csv_path is not None:
                        import csv

                        # load optional train stems to restrict to training split
                        train_set = None
                        try:
                            ts = Path(getattr(config, "train_source", ""))
                            if ts and ts.exists():
                                with open(ts, "r") as f:
                                    train_set = set(
                                        line.strip() for line in f if line.strip()
                                    )
                        except Exception:
                            train_set = None

                        pos_pixels = 0
                        tile_count = 0
                        with open(csv_path, newline="") as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                # column expected: 'rgb_tile' and 'positive_count'
                                rgb_name = (
                                    row.get("rgb_tile")
                                    or row.get("image")
                                    or row.get("image_base")
                                )
                                pos_cnt = (
                                    row.get("positive_count")
                                    or row.get("positive_pixels")
                                    or row.get("positive")
                                )
                                try:
                                    pos_val = (
                                        int(float(pos_cnt))
                                        if pos_cnt not in (None, "")
                                        else 0
                                    )
                                except Exception:
                                    pos_val = 0

                                # derive stem (without extension) to compare with train list
                                stem = None
                                if isinstance(rgb_name, str):
                                    stem = Path(rgb_name).stem

                                if train_set is not None:
                                    if stem is None or stem not in train_set:
                                        continue

                                pos_pixels += pos_val
                                tile_count += 1

                        if tile_count > 0:
                            # assume uniform tile size; compute total pixels
                            pixels_per_tile = int(
                                getattr(config, "image_height", 512)
                            ) * int(getattr(config, "image_width", 512))
                            total_pixels = float(tile_count * pixels_per_tile)
                            neg_pixels = max(total_pixels - float(pos_pixels), 1.0)
                            pos_pixels = max(float(pos_pixels), 1.0)

                            w_neg = total_pixels / neg_pixels
                            w_pos = total_pixels / pos_pixels

                            # normalize mean to 1 to keep CE scale stable
                            mean_w = (w_neg + w_pos) / 2.0
                            # clip extreme weights
                            max_w = float(getattr(config, "max_class_weight", 100.0))
                            w_neg = min(w_neg / mean_w, max_w)
                            w_pos = min(w_pos / mean_w, max_w)
                            weight = [w_neg, w_pos]
                except Exception:
                    weight = None

            # If pixel weights not produced, fall back to bucket-count weighting
            if weight is None and getattr(config, "oversample_positives", False):
                # Try bucket-based weighting
                try:
                    buckets_dir = Path(getattr(config, "stratified_buckets_dir", ""))
                    if buckets_dir and buckets_dir.exists():

                        def read_count(name):
                            p = buckets_dir / name
                            if not p.exists():
                                return 0
                            with open(p, "r") as f:
                                return sum(1 for ln in f if ln.strip())

                        neg_n = read_count("neg.txt")
                        small_n = read_count("small_pos.txt")
                        mid_n = read_count("mid_pos.txt")
                        full_n = read_count("full_pos.txt")
                        pos_n = small_n + mid_n + full_n

                        if neg_n > 0 and pos_n > 0:
                            total = float(neg_n + pos_n)
                            # inverse-frequency weighting
                            w_neg = total / float(neg_n)
                            w_pos = total / float(pos_n)
                            # normalize to mean=1 to keep CE scale stable (matching prior behaviour)
                            mean_w = (w_neg + w_pos) / 2.0
                            weight = [w_neg / mean_w, w_pos / mean_w]
                except Exception:
                    weight = None

                # Fallback: scalar oversample factor (kept for backward compatibility)
                if weight is None:
                    pos_factor = float(
                        getattr(config, "positive_oversample_factor", 1.0)
                    )
                    raw = [1.0, float(pos_factor)]
                    mean_w = float(sum(raw)) / len(raw)
                    weight = [w / mean_w for w in raw]
        except Exception:
            weight = None

        # Clamp computed class weights to configured min/max to avoid extremes
        if weight is not None:
            try:
                max_w = float(getattr(config, "max_class_weight", 100.0))
                min_w = float(getattr(config, "min_class_weight", 0.0))
                # ensure sensible ordering
                if min_w > max_w:
                    min_w = max_w

                weight = [float(weight[0]), float(weight[1])]
                weight[0] = min(max(weight[0], min_w), max_w)
                weight[1] = min(max(weight[1], min_w), max_w)
            except Exception:
                pass

        criterion = DiceCrossEntropyLoss(
            dice_weight=config.dice_weight,
            ce_weight=config.ce_weight,
            ignore_index=255,
            weight=weight,
        )
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=255)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params_list,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "SGDM":
        optimizer = torch.optim.SGD(
            params_list,
            lr=base_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch * config.warm_up_epoch,
    )

    def _move_optimizer_state(optimizer, device):
        # Move optimizer state tensors to the given device (in-place).
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    try:
                        state[k] = v.to(device)
                    except Exception:
                        # best-effort move; skip if it fails
                        pass

    # Register state and restore checkpoint (if any) while model/optimizer
    # parameters/optimizer states are still on CPU to avoid GPU spikes.
    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    if engine.distributed:
        logger.info(".............distributed training.............")
        if torch.cuda.is_available():
            model.cuda()
            # move optimizer state to the local GPU device
            _move_optimizer_state(optimizer, torch.device("cuda", engine.local_rank))
            model = DistributedDataParallel(
                model,
                device_ids=[engine.local_rank],
                output_device=engine.local_rank,
                find_unused_parameters=False,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # move model and optimizer state to chosen device
        model.to(device)
        _move_optimizer_state(optimizer, device)

    optimizer.zero_grad()
    model.train()
    logger.info("begin trainning:")

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        pbar = tqdm(
            range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format
        )
        dataloader = iter(train_loader)

        sum_loss = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]
            modal_xs = minibatch["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            aux_rate = 0.2
            loss = model(imgs, modal_xs, gts)

            # per-batch downweight if tile is overwhelmingly positive
            try:
                # exclude ignore_index (255) from the positive fraction calculation
                valid_mask = gts != 255
                valid_count = float(valid_mask.float().sum().item())
                if valid_count == 0:
                    pos_frac = 0.0
                else:
                    pos_frac = float(
                        ((gts > 0) & valid_mask).float().sum().item() / valid_count
                    )
            except Exception:
                pos_frac = 0.0
            if pos_frac >= config.full_pos_frac_threshold:
                down = float(config.full_pos_downweight)
                loss = loss * down

            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f"
                    % (reduce_loss.item(), (sum_loss / (idx + 1)))
                )
            else:
                sum_loss += loss.item()
                print_str = (
                    "Epoch {}/{}".format(epoch, config.nepochs)
                    + " Iter {}/{}:".format(idx + 1, config.niters_per_epoch)
                    + " lr=%.4e" % lr
                    + " loss=%.4f total_loss=%.4f"
                    % (loss.item(), (sum_loss / (idx + 1)))
                )

            del loss
            pbar.set_description(print_str, refresh=False)

        if (engine.distributed and (engine.local_rank == 0)) or (
            not engine.distributed
        ):
            tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)

        if (
            (epoch >= config.checkpoint_start_epoch)
            and (epoch % config.checkpoint_step == 0)
            or (epoch == config.nepochs)
        ):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(
                    config.checkpoint_dir, config.log_dir, config.log_dir_link
                )
            elif not engine.distributed:
                engine.save_and_link_checkpoint(
                    config.checkpoint_dir, config.log_dir, config.log_dir_link
                )
