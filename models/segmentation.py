"""
models/segmentation.py

A 4-channel (T1, T1ce, T2, FLAIR) 2D U-Net for brain-tumor segmentation, with a
NeuroMesh topology controller inserted at the bottleneck as a channel-level
"rewiring" gate intended to make the bottleneck representation more robust to
missing/corrupted input modalities or channel-wise activation failures.

SCOPE NOTE: this file implements an architecture only. It has not been trained
or validated on real BraTS data in this environment -- there is no BraTS data
here, and no network path to fetch any (this sandbox cannot reach
med.upenn.edu or kaggle.com). Any statement about this model's real-world
segmentation accuracy or robustness has to come from experiments *you* run on
properly licensed BraTS data, not from this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import NeuroMeshLayer


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class NeuroMeshUNet(nn.Module):
    """
    Args:
        in_channels: number of input MRI sequences (default 4: T1, T1ce, T2, FLAIR)
        num_classes: segmentation classes (default 4: background, necrotic/non-
                     enhancing core, edema, enhancing tumor -- the standard
                     BraTS label grouping after remapping label 4 -> 3)
        base_ch: base channel width for the U-Net
        controller_hidden: hidden width of the NeuroMesh controller
    """

    def __init__(self, in_channels=4, num_classes=4, base_ch=32, controller_hidden=128):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.inc = conv_block(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        bottleneck_ch = base_ch * 16
        self.controller = NeuroMeshLayer(feature_dim=bottleneck_ch, hidden_dim=controller_hidden)

        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, h_prev=None, failure_signal=None):
        """
        Args:
            x: [B, in_channels, H, W] -- dropped modalities should already be
               zeroed out by the data loader before this call.
            h_prev: optional controller hidden state carried across slices/timesteps.
            failure_signal: optional [B, bottleneck_ch] binary indicator; if not
               given, it's derived automatically from numerically-dead bottleneck channels.
        Returns:
            logits: [B, num_classes, H, W]
            h_new:  updated controller hidden state
            mask:   predicted bottleneck edge-activation mask (for loss terms / logging)
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck feature map [B, C, h, w]

        B, C, H, W = x5.shape
        pooled = F.adaptive_avg_pool2d(x5, 1).view(B, C)

        if failure_signal is None:
            failure_signal = (pooled.abs() < 1e-6).float()

        rewired, h_new, mask = self.controller(pooled, h_prev, failure_signal)

        gate = torch.sigmoid(rewired).view(B, C, 1, 1)  # channel-attention gate
        x5 = x5 * gate

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits, h_new, mask
