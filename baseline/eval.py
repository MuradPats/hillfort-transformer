import os
import cv2
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

# Baseline config (should expose the same fields your dataset/evaluator needs)
from baseline.config import config


# Reuse RGBX infrastructure
from RGBX.utils.pyt_utils import ensure_dir, parse_devices
from RGBX.utils.visualize import print_iou, show_img
from RGBX.engine.evaluator import Evaluator
from RGBX.engine.logger import get_logger
from RGBX.utils.metric import hist_info, compute_score
from RGBX.dataloader.RGBXDataset import RGBXDataset
from RGBX.utils.loss_opr import DiceCrossEntropyLoss
from RGBX.dataloader.dataloader import ValPre

# Baseline model
from baseline.models.unet_baseline_rgbx import UNetBaselineRGBX

logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data["data"]
        label = data["label"]
        modal_x = data["modal_x"]
        name = data["fn"]

        # Reuse the RGBX sliding-window eval helper from Evaluator.
        # Even if your baseline ignores modal_x, passing it is harmless.
        pred = self.sliding_eval_rgbX(
            img, modal_x, config.eval_crop_size, config.eval_stride_rate, device
        )

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {
            "hist": hist_tmp,
            "labeled": labeled_tmp,
            "correct": correct_tmp,
        }

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path + "_color")

            fn = name + ".png"

            # save colored result (palette)
            result_img = Image.fromarray(pred.astype(np.uint8), mode="P")
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path + "_color", fn))

            # save raw prediction
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info("Save the image " + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean, label, pred)
            cv2.imshow("comp_image", comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        for d in results:
            hist += d["hist"]
            correct += d["correct"]
            labeled += d["labeled"]

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
            hist, correct, labeled
        )
        result_line = print_iou(
            iou,
            freq_IoU,
            mean_pixel_acc,
            pixel_acc,
            dataset.class_names,
            show_no_back=False,
        )
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", default="last", type=str)
    parser.add_argument("-d", "--devices", default="0", type=str)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--show_image", "-s", default=False, action="store_true")
    parser.add_argument("--save_path", "-p", default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    # Keep the same criterion type as training (safe even if it has no parameters)
    weight = None
    try:
        if getattr(config, "oversample_positives", False):
            pos_factor = float(getattr(config, "positive_oversample_factor", 1.0))
            raw = [1.0, float(pos_factor)]
            mean_w = float(sum(raw)) / len(raw)
            weight = [w / mean_w for w in raw]
    except Exception:
        weight = None

    criterion = DiceCrossEntropyLoss(
        dice_weight=config.dice_weight,
        ce_weight=config.ce_weight,
        ignore_index=getattr(config, "ignore_index", 255),
        weight=weight,
    )

    # Baseline network (same forward signature as RGBX: (imgs, modal_x, gts?))
    network = UNetBaselineRGBX(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)

    # Dataset settings: reuse exactly the same keys your RGBXDataset expects.
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

    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, "val", val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(
            dataset,
            config.num_classes,
            config.norm_mean,
            config.norm_std,
            network,
            config.eval_scale_array,
            config.eval_flip,
            all_dev,
            args.verbose,
            args.save_path,
            args.show_image,
        )
        segmentor.run(
            config.checkpoint_dir,
            args.epochs,
            config.val_log_file,
            config.link_val_log_file,
        )
