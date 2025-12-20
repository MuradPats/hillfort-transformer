import cv2
import torch
import numpy as np
from torch.utils import data
import random
from config import config
from utils.transforms import (
    generate_random_crop_pos,
    random_crop_pad_to_shape,
    normalize,
)
from .stratified_sampler import StratifiedBatchSampler
from pathlib import Path
import os


def random_mirror(rgb, gt, modal_x):
    if random.random() >= 0.5:
        rgb = cv2.flip(rgb, 1)
        gt = cv2.flip(gt, 1)
        modal_x = cv2.flip(modal_x, 1)

    return rgb, gt, modal_x


def random_scale(rgb, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(rgb.shape[0] * scale)
    sw = int(rgb.shape[1] * scale)
    rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return rgb, gt, modal_x, scale


class TrainPre(object):
    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, rgb, gt, modal_x):
        rgb, gt, modal_x = random_mirror(rgb, gt, modal_x)
        if config.train_scale_array is not None:
            rgb, gt, modal_x, scale = random_scale(
                rgb, gt, modal_x, config.train_scale_array
            )

        rgb = normalize(rgb, self.norm_mean, self.norm_std)
        modal_x = normalize(modal_x, self.norm_mean, self.norm_std)

        crop_size = (config.image_height, config.image_width)
        crop_pos = generate_random_crop_pos(rgb.shape[:2], crop_size)

        p_rgb, _ = random_crop_pad_to_shape(rgb, crop_pos, crop_size, 0)
        p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        p_modal_x, _ = random_crop_pad_to_shape(modal_x, crop_pos, crop_size, 0)

        p_rgb = p_rgb.transpose(2, 0, 1)
        p_modal_x = p_modal_x.transpose(2, 0, 1)

        return p_rgb, p_gt, p_modal_x


class ValPre(object):
    def __call__(self, rgb, gt, modal_x):
        return rgb, gt, modal_x


def get_train_loader(engine, dataset):
    data_setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "class_names": config.class_names,
    }
    train_preprocess = TrainPre(config.norm_mean, config.norm_std)

    train_dataset = dataset(
        data_setting,
        "train",
        train_preprocess,
        config.batch_size * config.niters_per_epoch,
    )

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    # Distributed handling (keep existing behavior)
    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    # Stratified batch sampling: if config provides a buckets dir, build sampler
    strat_sampler = None
    if hasattr(config, "stratified_buckets_dir") and config.stratified_buckets_dir:
        buckets_dir = Path(config.stratified_buckets_dir)
        if buckets_dir.exists():
            # read bucket files
            def read_bucket(name):
                p = buckets_dir / name
                if not p.exists():
                    return []
                with open(p, "r") as f:
                    return [ln.strip() for ln in f if ln.strip()]

            neg = read_bucket("neg.txt")
            small = read_bucket("small_pos.txt")
            mid = read_bucket("mid_pos.txt")
            full = read_bucket("full_pos.txt")

            # map stems to indices using dataset helper
            def stems_to_indices(stems):
                idxs = []
                for s in stems:
                    i = train_dataset.stem_to_index(s)
                    if i is not None:
                        idxs.append(i)
                return idxs

            neg_idx = stems_to_indices(neg)
            small_idx = stems_to_indices(small)
            mid_idx = stems_to_indices(mid)
            full_idx = stems_to_indices(full)

            # If running distributed, split each bucket among ranks to avoid duplication
            if engine.distributed:
                try:
                    world_size = engine.world_size
                    rank = engine.rank
                except Exception:
                    import torch.distributed as dist

                    world_size = dist.get_world_size()
                    rank = dist.get_rank()

                def shard_list(lst, r, ws):
                    if not lst:
                        return []
                    return lst[r::ws]

                neg_idx = shard_list(neg_idx, rank, world_size)
                small_idx = shard_list(small_idx, rank, world_size)
                mid_idx = shard_list(mid_idx, rank, world_size)
                full_idx = shard_list(full_idx, rank, world_size)

            # compute per-batch counts from proportions if provided
            if (
                hasattr(config, "stratified_proportions")
                and config.stratified_proportions
            ):
                props = config.stratified_proportions
            else:
                props = [0.5, 0.4, 0.09, 0.01]

            counts = [int(p * batch_size) for p in props]
            # adjust to match batch_size
            diff = batch_size - sum(counts)
            i = 0
            while diff != 0:
                counts[i % 4] += 1 if diff > 0 else -1
                diff = batch_size - sum(counts)
                i += 1

            strat_sampler = StratifiedBatchSampler(
                buckets=[neg_idx, small_idx, mid_idx, full_idx],
                counts=counts,
                epoch_size=config.niters_per_epoch,
                replace=True,
                shuffle=True,
                seed=config.seed,
            )

    # prefer strat_sampler if created; otherwise use existing train_sampler
    if strat_sampler is not None:
        # strat_sampler is a BatchSampler; pass it as `batch_sampler`
        train_loader = data.DataLoader(
            train_dataset,
            batch_sampler=strat_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=config.num_workers,
            drop_last=True,
            shuffle=is_shuffle,
            pin_memory=True,
            sampler=train_sampler,
        )

    return train_loader, train_sampler
