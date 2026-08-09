"""
utils/evaluation.py

Evaluation utilities that measure:
  1. Per-class and mean Dice as a function of modality-dropout condition
     (e.g. always zero the FLAIR channel, or randomly zero channels at some rate).
  2. Wall-clock latency of the model's forward pass, measured directly with
     torch timers (CUDA-synchronized when run on GPU).

This module does NOT hard-code any target Dice score or latency figure. Every
number it returns is measured live, on whatever model checkpoint and hardware
you run it with -- including the mock/synthetic pipeline in train.py, whose
numbers are illustrative of the *pipeline*, not of clinical performance.
"""

import time
import torch
import torch.nn.functional as F


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
