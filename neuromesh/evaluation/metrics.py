"""
utils/evaluation.py

Evaluation utilities that measure:
  1. Per-class and mean Dice as a function of modality-dropout condition
     (e.g. always zero the FLAIR channel, or randomly zero channels at some rate).
  2. Wall-clock latency of the model's forward pass, measured directly with
     torch timers (CUDA-synchronized when run on GPU).
  3. Full-volume BraTS composite-region metrics (WT/TC/ET Dice, HD95,
     sensitivity, precision) -- reconstructed per-patient from per-slice
     predictions, matching how the BraTS challenge itself scores submissions
     (volumetric, not per-slice-averaged).

This module does NOT hard-code any target Dice score or latency figure. Every
number it returns is measured live, on whatever model checkpoint and hardware
you run it with -- including the mock/synthetic pipeline in train.py, whose
numbers are illustrative of the *pipeline*, not of clinical performance.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion


@torch.no_grad()
def dice_per_class(logits, targets, num_classes, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        target_c = (targets == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dices.append(((2 * intersection + eps) / (union + eps)).item())
    return dices


# Label convention used throughout this repo (see H5SliceBraTSDataset
# docstring in data/brats_loader.py): 0=background, 1=necrotic/non-enhancing
# core, 2=edema, 3=enhancing tumor. These composite regions are what BraTS
# itself scores, not the raw 4-way label map.
REGION_LABEL_SETS = {
    "WT": {1, 2, 3},  # whole tumor: any tumor label
    "TC": {1, 3},      # tumor core: necrotic + enhancing (excludes edema)
    "ET": {3},         # enhancing tumor only
}


def _binary_region(label_volume, label_set):
    mask = np.zeros(label_volume.shape, dtype=bool)
    for lbl in label_set:
        mask |= (label_volume == lbl)
    return mask


def _dice_binary(pred_mask, target_mask, eps=1e-6):
    inter = np.logical_and(pred_mask, target_mask).sum()
    union = pred_mask.sum() + target_mask.sum()
    return float((2 * inter + eps) / (union + eps))


def _hd95_binary(pred_mask, target_mask, spacing=(1.0, 1.0, 1.0)):
    """95th-percentile symmetric surface (Hausdorff) distance, via boundary
    erosion + Euclidean distance transform -- the standard approach (same
    method used by medpy/nnU-Net-style implementations), self-contained here
    on scipy rather than adding medpy as a dependency.

    Returns None (not a fabricated number) when either mask is empty --
    HD95 is undefined if there's no surface to measure, and a placeholder
    like 0 or inf would misrepresent that as a real answer both directions."""
    if pred_mask.sum() == 0 and target_mask.sum() == 0:
        return None  # both empty: nothing to measure, not "perfect" (0)
    if pred_mask.sum() == 0 or target_mask.sum() == 0:
        return None  # one empty, one not: undefined surface distance, not "infinite"

    pred_border = pred_mask & ~binary_erosion(pred_mask)
    target_border = target_mask & ~binary_erosion(target_mask)
    if pred_border.sum() == 0 or target_border.sum() == 0:
        return None

    target_dt = distance_transform_edt(~target_mask, sampling=spacing)
    pred_dt = distance_transform_edt(~pred_mask, sampling=spacing)

    d_pred_to_target = target_dt[pred_border]
    d_target_to_pred = pred_dt[target_border]
    all_d = np.concatenate([d_pred_to_target, d_target_to_pred])
    return float(np.percentile(all_d, 95))


def _sensitivity_precision_binary(pred_mask, target_mask, eps=1e-6):
    tp = np.logical_and(pred_mask, target_mask).sum()
    fn = np.logical_and(~pred_mask, target_mask).sum()
    fp = np.logical_and(pred_mask, ~target_mask).sum()
    # If the region is absent from ground truth entirely, sensitivity is
    # undefined (0/0) -- report None rather than a fabricated 1.0 or 0.0.
    sensitivity = None if (tp + fn) == 0 else float(tp / (tp + fn + eps))
    precision = None if (tp + fp) == 0 else float(tp / (tp + fp + eps))
    return sensitivity, precision


def compute_patient_region_metrics(pred_volume, target_volume, spacing=(1.0, 1.0, 1.0)):
    """
    Full-volume (not per-slice-averaged) WT/TC/ET Dice, HD95, sensitivity,
    and precision for ONE patient. pred_volume / target_volume are integer
    label arrays of identical shape [n_slices, H, W] with labels
    {0,1,2,3} per this repo's convention.

    spacing: physical voxel spacing (z, y, x) in mm, used for HD95. BraTS's
    standard preprocessing resamples all volumes to 1mm^3 isotropic
    resolution -- this is assumed here (not independently re-verified from
    this particular Kaggle mirror's files, which don't carry NIfTI affine
    metadata to check directly). Report this assumption alongside any HD95
    number in the manuscript.
    """
    out = {}
    for region, label_set in REGION_LABEL_SETS.items():
        pred_mask = _binary_region(pred_volume, label_set)
        target_mask = _binary_region(target_volume, label_set)
        sens, prec = _sensitivity_precision_binary(pred_mask, target_mask)
        out[region] = {
            "dice": _dice_binary(pred_mask, target_mask),
            "hd95_mm": _hd95_binary(pred_mask, target_mask, spacing=spacing),
            "sensitivity": sens,
            "precision": prec,
            "pred_voxel_count": int(pred_mask.sum()),
            "target_voxel_count": int(target_mask.sum()),
        }
    return out


def apply_modality_dropout(x, modality_idx=None, drop_prob=0.0, generator=None):
    """
    Zero out one or more of the 4 input MRI channels.

    Args:
        x: [B, 4, H, W]
        modality_idx: int, list[int], or None.
            If given, ALWAYS zero these channel indices (e.g. 3 == FLAIR).
        drop_prob: if modality_idx is None, each channel is independently
            zeroed with this probability.
    """
    x = x.clone()
    if modality_idx is not None:
        idxs = [modality_idx] if isinstance(modality_idx, int) else list(modality_idx)
        x[:, idxs, :, :] = 0.0
        return x
    if drop_prob > 0:
        keep = (torch.rand(x.size(0), x.size(1), generator=generator) > drop_prob).float()
        keep = keep.to(x.device).view(x.size(0), x.size(1), 1, 1)
        x = x * keep
    return x


@torch.no_grad()
def evaluate_modality_dropout(model, dataloader, num_classes, device,
                               dropout_configs=None, max_batches=None):
    """
    Sweep a list of dropout configurations and report measured Dice + latency.

    dropout_configs: list of dicts, e.g.
        [{"name": "clean",         "modality_idx": None, "drop_prob": 0.0},
         {"name": "flair_dropped", "modality_idx": 3,    "drop_prob": 0.0},
         {"name": "random_20pct",  "modality_idx": None, "drop_prob": 0.2}]
    """
    if dropout_configs is None:
        dropout_configs = [
            {"name": "clean", "modality_idx": None, "drop_prob": 0.0},
            {"name": "flair_dropped", "modality_idx": 3, "drop_prob": 0.0},
        ]

    model.eval()
    results = {}

    for cfg in dropout_configs:
        all_dices, latencies_ms = [], []
        h_state = None

        for i, (x, y) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            x_corrupted = apply_modality_dropout(x, cfg.get("modality_idx"), cfg.get("drop_prob", 0.0))

            # The controller's recurrent hidden state is carried across
            # batches within a sweep. On real (non-toy) data the final batch
            # of an epoch is frequently smaller than the rest (dataset size
            # not evenly divisible by batch_size), which otherwise crashes
            # GRUCell with a batch-size mismatch. Synthetic defaults elsewhere
            # in this repo (e.g. n_train=32, batch_size=4) always divide
            # evenly and never hit this path -- real data reliably does.
            if h_state is not None and h_state.size(0) != x_corrupted.size(0):
                h_state = None

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            logits, h_state, _ = model(x_corrupted, h_state)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)
            all_dices.append(dice_per_class(logits, y, num_classes))

        mean_dice_per_class = torch.tensor(all_dices).mean(dim=0).tolist()
        results[cfg["name"]] = {
            "mean_dice_per_class": mean_dice_per_class,
            "mean_dice": float(sum(mean_dice_per_class) / len(mean_dice_per_class)),
            "mean_latency_ms": float(sum(latencies_ms) / len(latencies_ms)),
            "max_latency_ms": float(max(latencies_ms)),
        }

    return results


@torch.no_grad()
def benchmark_recovery_latency(model, sample_batch, fault_counts, device, n_repeats=10):
    """
    Measures END-TO-END forward latency as a function of a simulated
    "fault count" -- here operationalized as the number of independent
    modality-dropout / channel-zeroing events applied within one batch before
    the forward pass. This produces REAL measured numbers on this hardware;
    it is not the same benchmark as the draft manuscript's Table 7 / Table D.3,
    which describe latency broken down by internal detect/GNN/mask/re-exec
    stages that this simplified controller does not separately instrument.

    Returns: dict fault_count -> {"mean_ms": ..., "std_ms": ...}
    """
    model.eval()
    x, _ = sample_batch
    x = x.to(device)
    results = {}

    for n_faults in fault_counts:
        timings = []
        for _ in range(n_repeats):
            x_faulted = x.clone()
            B, C = x_faulted.size(0), x_faulted.size(1)
            for _ in range(n_faults):
                b = torch.randint(0, B, (1,)).item()
                c = torch.randint(0, C, (1,)).item()
                x_faulted[b, c] = 0.0

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x_faulted)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

        timings_t = torch.tensor(timings)
        results[n_faults] = {"mean_ms": timings_t.mean().item(), "std_ms": timings_t.std().item()}

    return results
