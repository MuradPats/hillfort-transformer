import os
import sys
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

# Baseline config
from baseline.config import config
# If baseline/config.py is plain module vars (like above), replace with:
# import config as config

# Reuse RGBX infrastructure (no duplication)
from RGBX.dataloader.dataloader import get_train_loader
from RGBX.dataloader.RGBXDataset import RGBXDataset
from RGBX.utils.lr_policy import WarmUpPolyLR
from RGBX.engine.engine import Engine
from RGBX.engine.logger import get_logger
from RGBX.utils.pyt_utils import all_reduce_tensor
from RGBX.utils.loss_opr import DiceCrossEntropyLoss

# Baseline model
from baseline.models import UNetBaselineRGBX

from tensorboardX import SummaryWriter


def build_criterion(cfg):
    # same as your RGBX train.py logic, simplified to “use same loss”
    weight = None
    # (optional) you can copy-paste your weight computation block here later
    return DiceCrossEntropyLoss(
        dice_weight=cfg.dice_weight,
        ce_weight=cfg.ce_weight,
        ignore_index=getattr(cfg, "ignore_index", 255),
        weight=weight,
    )
def main():
    parser = argparse.ArgumentParser()
    logger = get_logger()

    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "169710")


    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        cudnn.benchmark = True

        seed = getattr(config, "seed", 123)
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # dataloader (reuse RGBX)
        train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

        # tensorboard
        if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
            tb_dir = config.tb_dir + "/{}".format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + "/tb"
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        # loss + model
        criterion = build_criterion(config)

        if engine.distributed:
            BatchNorm2d = nn.SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d

        model = UNetBaselineRGBX(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)

        # optimizer
        base_lr = config.lr
        params_list = [{"params": model.parameters(), "lr": base_lr}]

        if config.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
            )
        elif config.optimizer == "SGDM":
            optimizer = torch.optim.SGD(
                params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay
            )
        else:
            raise NotImplementedError

        # lr schedule
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(
            base_lr,
            config.lr_power,
            total_iteration,
            config.niters_per_epoch * config.warm_up_epoch,
        )

        # device / ddp
        if engine.distributed:
            logger.info(".............distributed training.............")
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(
                    model,
                    device_ids=[engine.local_rank],
                    output_device=engine.local_rank,
                    find_unused_parameters=False,
                )
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()
        logger.info("begin training baseline UNet:")

        for epoch in range(engine.state.epoch, config.nepochs + 1):
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
            dataloader = iter(train_loader)

            sum_loss = 0.0

            for idx in pbar:
                engine.update_iteration(epoch, idx)

                minibatch = next(dataloader)
                imgs = minibatch["data"].cuda(non_blocking=True)
                gts = minibatch["label"].cuda(non_blocking=True)
                modal_xs = minibatch.get("modal_x", None)
                if modal_xs is not None:
                    modal_xs = modal_xs.cuda(non_blocking=True)

                loss = model(imgs, modal_xs, gts)

                # per-batch downweight if tile is overwhelmingly positive (same as your script)
                try:
                    valid_mask = gts != getattr(config, "ignore_index", 255)
                    valid_count = float(valid_mask.float().sum().item())
                    if valid_count == 0:
                        pos_frac = 0.0
                    else:
                        pos_frac = float(((gts > 0) & valid_mask).float().sum().item() / valid_count)
                except Exception:
                    pos_frac = 0.0

                if pos_frac >= config.full_pos_frac_threshold:
                    loss = loss * float(config.full_pos_downweight)

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
                        f"Epoch {epoch}/{config.nepochs} "
                        f"Iter {idx+1}/{config.niters_per_epoch}: "
                        f"lr={lr:.4e} loss={reduce_loss.item():.4f} total_loss={(sum_loss/(idx+1)):.4f}"
                    )
                else:
                    sum_loss += loss.item()
                    print_str = (
                        f"Epoch {epoch}/{config.nepochs} "
                        f"Iter {idx+1}/{config.niters_per_epoch}: "
                        f"lr={lr:.4e} loss={loss.item():.4f} total_loss={(sum_loss/(idx+1)):.4f}"
                    )

                del loss
                pbar.set_description(print_str, refresh=False)

            if (engine.distributed and engine.local_rank == 0) or (not engine.distributed):
                tb.add_scalar("train_loss", sum_loss / len(pbar), epoch)

            if (
                (epoch >= config.checkpoint_start_epoch and epoch % config.checkpoint_step == 0)
                or (epoch == config.nepochs)
            ):
                if (not engine.distributed) or (engine.distributed and engine.local_rank == 0):
                    engine.save_and_link_checkpoint(
                        config.checkpoint_dir, config.log_dir, config.log_dir_link
                    )
if __name__ == "__main__":
    main()