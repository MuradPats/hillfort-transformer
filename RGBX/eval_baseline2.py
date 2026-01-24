import argparse
import os

import torch
import torch.nn as nn

from config import config
from utils.pyt_utils import parse_devices, load_model, link_file
from RGBX.eval import SegEvaluator
from engine.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from dataloader.dataloader import ValPre

# baseline2 import
from baseline2.model import UNetSmall

logger = get_logger()


class BaselineAdapter(nn.Module):
    """Adapter that makes a baseline UNetSmall callable like an RGBX network.

    The RGBX evaluator calls the network as `model(input_rgb, input_x)`.
    This adapter concatenates modality `input_x` to `input_rgb` when present
    so the baseline UNet receives the expected channel count.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, img, modal_x=None):
        # img: (B,C,H,W), modal_x: (B,C2,H,W) or (B,1,H,W)
        if modal_x is None:
            return self.base(img)

        # Ensure tensors are compatible and concatenate on channel dim
        if img.dim() == 4 and modal_x.dim() == 4:
            try:
                inp = torch.cat([img, modal_x], dim=1)
            except Exception:
                # fallback: ignore modal_x if concat fails
                inp = img
        else:
            inp = img

        return self.base(inp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", required=True, help="Path to baseline2 checkpoint (.pth)"
    )
    parser.add_argument("-d", "--devices", default="0", type=str)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--show_image", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    # Load checkpoint first so we can infer the trained model's input channels.
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    def infer_in_channels_from_state(sd: dict) -> int:
        # Heuristic: find the first 4-D tensor (conv weight) and return its in_channels
        for k in sorted(sd.keys()):
            v = sd[k]
            if hasattr(v, "ndim") and v.ndim == 4:
                return int(v.shape[1])
        return None

    inferred = infer_in_channels_from_state(state)
    if inferred is not None:
        in_ch = inferred
        logger.info(f"Inferred baseline input channels from checkpoint: {in_ch}")
    else:
        # Fallback to RGBX config: assume RGB + single-channel X if x_root present
        modal_ch = 1 if getattr(config, "x_is_single_channel", False) else 1
        in_ch = 3 + (modal_ch if getattr(config, "x_root_folder", None) else 0)
        logger.info(f"Could not infer channels; falling back to in_ch={in_ch}")

    base = UNetSmall(in_channels=in_ch, num_classes=config.num_classes)

    # Load checkpoint into baseline model (best-effort)
    try:
        base.load_state_dict(state, strict=False)
    except Exception:
        logger.warning("Direct load into baseline failed; attempting alternative keys")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            try:
                base.load_state_dict(ckpt["state_dict"], strict=False)
            except Exception:
                logger.warning(
                    "Fallback state_dict load failed; continuing with partially loaded model"
                )

    network = BaselineAdapter(base)

    # Build dataset using RGBX config so evaluator uses same preprocessing
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
    }

    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, "val", val_pre)

    # Instantiate SegEvaluator with the adapted baseline network
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

    # If the user passed a checkpoint file path, load it directly to avoid
    # `Evaluator.run` mis-parsing paths that contain hyphens.
    if os.path.isfile(args.ckpt):
        logger.info(f"Loading checkpoint file directly: {args.ckpt}")
        segmentor.val_func = load_model(network, args.ckpt)

        if len(all_dev) == 1:
            result_line = segmentor.single_process_evalutation()
        else:
            result_line = segmentor.multi_process_evaluation()

        # Write/Link log as Evaluator.run would
        with open(config.val_log_file, "a") as results:
            results.write("Model: " + args.ckpt + "\n")
            results.write(result_line + "\n")
            results.flush()
        try:
            link_file(config.val_log_file, config.link_val_log_file)
        except Exception:
            logger.warning("Could not create link for val log file")
    else:
        # Fall back to the Evaluator.run logic (accepts epoch strings, ranges, or .pth names)
        model_path = "."
        model_indice = args.ckpt
        segmentor.run(
            model_path, model_indice, config.val_log_file, config.link_val_log_file
        )


if __name__ == "__main__":
    main()
