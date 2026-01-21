import torch
import torch.nn as nn
from .unet_baseline import UNetBaseline


class UNetBaselineRGBX(nn.Module):
    """
    Wrapper that optionally concatenates modal_x to RGB.
    Keeps forward signature: forward(imgs, modal_xs, gts) -> loss
    """
    def __init__(self, cfg, criterion, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cfg = cfg
        self.criterion = criterion

        use_modal_x = bool(getattr(cfg, "baseline_use_modal_x", False))
        modal_ch = int(getattr(cfg, "modal_x_channels", 1))  # set this correctly!
        in_ch = 3 + (modal_ch if use_modal_x else 0)

        self.use_modal_x = use_modal_x
        self.net = UNetBaseline(
            in_channels=in_ch,
            num_classes=int(getattr(cfg, "num_classes", 2)),
            base_channels=int(getattr(cfg, "baseline_base_channels", 64)),
            bilinear=bool(getattr(cfg, "baseline_bilinear", True)),
            norm_layer=norm_layer,
            criterion=criterion,
        )

    def forward(self, imgs, modal_xs=None, gts=None):
        x = imgs
        if self.use_modal_x:
            if modal_xs is None:
                raise ValueError("baseline_use_modal_x=True but modal_xs is None.")
            # modal_xs should be [B, modal_ch, H, W]
            x = torch.cat([imgs, modal_xs], dim=1)

        return self.net(x, None, gts)
