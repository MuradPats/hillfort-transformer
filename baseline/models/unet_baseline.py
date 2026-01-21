import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, norm_layer=norm_layer),
            ConvBNReLU(out_ch, out_ch, norm_layer=norm_layer),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, norm_layer=norm_layer)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Upsample + concat skip + DoubleConv.
    Uses bilinear upsample by default (more stable baseline than transposed conv).
    """
    def __init__(self, in_ch, skip_ch, out_ch, norm_layer=nn.BatchNorm2d, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            mid_ch = in_ch
        else:
            # transposed conv reduces channels
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            mid_ch = in_ch // 2

        self.conv = DoubleConv(mid_ch + skip_ch, out_ch, norm_layer=norm_layer)

    def forward(self, x, skip):
        x = self.up(x)
        # handle odd input shapes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    """
    UNet segmentation baseline.
    - If criterion is provided, forward returns loss like your current model.
    - Otherwise returns logits.

    Expected labels:
      - binary (background=0, hillfort=1) with ignore_index=255 allowed
      - criterion should handle ignore_index
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        base_channels: int = 64,
        bilinear: bool = True,
        norm_layer=nn.BatchNorm2d,
        criterion=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = criterion

        c = base_channels
        self.inc = DoubleConv(in_channels, c, norm_layer=norm_layer)
        self.down1 = Down(c, c * 2, norm_layer=norm_layer)
        self.down2 = Down(c * 2, c * 4, norm_layer=norm_layer)
        self.down3 = Down(c * 4, c * 8, norm_layer=norm_layer)
        self.down4 = Down(c * 8, c * 16, norm_layer=norm_layer)

        self.up1 = Up(c * 16, c * 8, c * 8, norm_layer=norm_layer, bilinear=bilinear)
        self.up2 = Up(c * 8, c * 4, c * 4, norm_layer=norm_layer, bilinear=bilinear)
        self.up3 = Up(c * 4, c * 2, c * 2, norm_layer=norm_layer, bilinear=bilinear)
        self.up4 = Up(c * 2, c, c, norm_layer=norm_layer, bilinear=bilinear)

        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, imgs, modal_xs=None, gts=None):
        # imgs: [B, C, H, W]
        x = imgs

        x1 = self.inc(x)       # c
        x2 = self.down1(x1)    # 2c
        x3 = self.down2(x2)    # 4c
        x4 = self.down3(x3)    # 8c
        x5 = self.down4(x4)    # 16c

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        if (self.criterion is not None) and (gts is not None):
            return self.criterion(logits, gts)
        return logits
