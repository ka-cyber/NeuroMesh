"""
models/baselines.py

The original internal draft's Table 1 claimed NeuroMesh outperforms "Dropout,"
"Ensemble," "DynMoE," and "Rerouting" baselines -- but none of those baselines
were ever actually implemented anywhere in this project. Numbers can't be
fairly compared against baselines that don't exist as code. This file
implements the two most standard, unambiguous ones so a real comparison is
possible once real training happens. DynMoE and "Rerouting" were never
precisely specified in the original draft (no architecture, no
hyperparameters) -- implementing them would mean inventing an architecture
and attributing it to the draft, which isn't done here. If you want those
comparisons, they need to be specified and implemented deliberately, ideally
against a published DynMoE reference implementation.

Both baselines share the same U-Net backbone (models/segmentation.py's
conv_block/Down/Up) so that any measured difference is attributable to the
robustness mechanism, not to backbone capacity differences.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuromesh.models.neuromesh import conv_block, Down, Up


class DropoutUNet(nn.Module):
    """
    Standard U-Net with spatial dropout at the bottleneck (and optionally after
    each encoder stage), left ACTIVE at inference time (MC-Dropout style) so
    that, unlike training-only dropout, it provides some test-time stochastic
    regularization under corrupted input -- this is the fairest "Dropout"
    baseline to compare a test-time-adaptive method like NeuroMesh against.
    """

    def __init__(self, in_channels=4, num_classes=4, base_ch=32, drop_prob=0.5):
        super().__init__()
        self.inc = conv_block(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        self.bottleneck_dropout = nn.Dropout2d(p=drop_prob)
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, mc_dropout_at_inference=True):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if mc_dropout_at_inference:
            was_training = self.bottleneck_dropout.training
            self.bottleneck_dropout.train()  # force dropout active even in .eval() mode
            x5 = self.bottleneck_dropout(x5)
            self.bottleneck_dropout.train(was_training)
        else:
            x5 = self.bottleneck_dropout(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class EnsembleUNet(nn.Module):
    """
    A fixed-size ensemble of independently-initialized U-Nets (no NeuroMesh
    controller), averaging softmax probabilities at inference. This is the
    standard ensembling baseline: no input-adaptive behavior, but redundant
    capacity. Compute cost scales linearly with `n_models` -- report this
    honestly alongside any accuracy comparison (an n=3 ensemble uses ~3x the
    FLOPs and ~3x the latency of a single backbone, before any NeuroMesh
    controller overhead is added).
    """

    def __init__(self, in_channels=4, num_classes=4, base_ch=32, n_models=3, seed=0):
        super().__init__()
        self.n_models = n_models
        models = []
        for i in range(n_models):
            torch.manual_seed(seed + i)
            models.append(_PlainUNet(in_channels, num_classes, base_ch))
        self.models = nn.ModuleList(models)

    def forward(self, x):
        probs = [F.softmax(m(x), dim=1) for m in self.models]
        mean_probs = torch.stack(probs, dim=0).mean(dim=0)
        # Return log-probabilities as "logits" so downstream CE/argmax code
        # written against raw logits still behaves sensibly.
        return torch.log(mean_probs.clamp(min=1e-8))


class _PlainUNet(nn.Module):
    """Plain U-Net, no controller, no dropout -- one ensemble member."""

    def __init__(self, in_channels=4, num_classes=4, base_ch=32):
        super().__init__()
        self.inc = conv_block(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class StaticGatedUNet(nn.Module):
    """
    Control for the mechanism-analysis finding that NeuroMeshUNet's
    controller mask isn't actually differentiating by condition in this
    pilot (mean|mask_clean - mask_failure| ~ 0.00000 at both 4 and 8 epochs,
    val set). If the mask isn't adapting, NeuroMesh's TC/ET advantage over
    plain U-Net might come from nothing more than "a learned channel gate
    exists at the bottleneck at all" -- not from anything about the
    controller's specific machinery (GRU hidden state, GCN/adjacency mask,
    failure-signal conditioning). This model isolates that: same backbone,
    same bottleneck channel-gating OPERATION (x5 = x5 * sigmoid(gate)) as
    NeuroMeshUNet, but the gate comes from a plain feedforward
    squeeze-and-excitation-style MLP on the pooled bottleneck vector --
    no recurrence, no failure signal, no adjacency mask, no conditioning on
    which (if any) modality is missing. It is STATIC in the same sense the
    measured NeuroMesh mask turned out to be static, not by design
    limitation but by direct analogy to the empirical finding.

    NOT parameter-matched to NeuroMeshLayer as a whole -- most of that
    module's parameters live in the mask MLP that projects to a
    [feature_dim, feature_dim] edge matrix (65,536 entries at base_ch=16),
    which is exactly the component this control deliberately excludes, since
    testing whether that specific machinery matters is the point of running
    it. Report both parameter counts explicitly (see train_real.py's startup
    log) rather than claim equivalence.
    """

    def __init__(self, in_channels=4, num_classes=4, base_ch=32, gate_hidden=64):
        super().__init__()
        self.inc = conv_block(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)
        bottleneck_ch = base_ch * 16
        self.gate_mlp = nn.Sequential(
            nn.Linear(bottleneck_ch, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, bottleneck_ch),
        )
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch)
        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, return_debug=False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        B, C, H, W = x5.shape
        pooled = F.adaptive_avg_pool2d(x5, 1).view(B, C)
        gate = torch.sigmoid(self.gate_mlp(pooled)).view(B, C, 1, 1)
        x5 = x5 * gate

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if return_debug:
            return logits, {"gate": gate.view(B, C).detach()}
        return logits
