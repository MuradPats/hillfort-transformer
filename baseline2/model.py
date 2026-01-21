from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# VRAM debug toggle for model internals
DEBUG_VRAM = os.getenv("DEBUG_VRAM", "0") == "1"

def _show_mem(tag: str = "") -> None:
    if not DEBUG_VRAM:
        return
    if not torch.cuda.is_available():
        print(f"[model VRAM] {tag}: cuda not available")
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() // (1024 ** 2)
    reserved = torch.cuda.memory_reserved() // (1024 ** 2)
    print(f"[model VRAM] {tag}: allocated={alloc}MiB reserved={reserved}MiB")

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        # encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bott = ConvBlock(128, 256)

        # decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if DEBUG_VRAM:
            _show_mem("forward_start")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bott(self.pool(e3))

        d3 = self.up3(b)
        if DEBUG_VRAM:
            print("concat d3/e3 shapes:", d3.shape, e3.shape)
            _show_mem("before_concat_d3")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        if DEBUG_VRAM:
            _show_mem("after_concat_d3")

        d2 = self.up2(d3)
        if DEBUG_VRAM:
            print("concat d2/e2 shapes:", d2.shape, e2.shape)
            _show_mem("before_concat_d2")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        if DEBUG_VRAM:
            _show_mem("after_concat_d2")

        d1 = self.up1(d2)
        if DEBUG_VRAM:
            print("concat d1/e1 shapes:", d1.shape, e1.shape)
            _show_mem("before_concat_d1")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        if DEBUG_VRAM:
            _show_mem("after_concat_d1")

        return self.head(d1)
