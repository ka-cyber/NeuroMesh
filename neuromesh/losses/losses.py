"""
utils/loss.py

Composite training objective:

    L_total = L_task + lambda1 * L_rewire + lambda2 * L_sparsity + lambda3 * L_diversity

  L_task      : Dice + Cross-Entropy segmentation loss on clean input.
  L_rewire    : KL divergence between predictions on clean input and predictions
                on modality-corrupted input -- pushes the controller to find
                alternate pathways that preserve the output distribution.
  L_sparsity  : fraction-of-active-edges penalty on the predicted mask(s).
  L_diversity : pairwise cosine-similarity hinge penalty across masks, to
                discourage a degenerate, non-adaptive policy.

The lambda defaults below (0.1 / 0.01 / 0.001) are simply carried over from the
draft manuscript's *stated* defaults -- they are starting points, not validated
hyperparameters. Re-tune them against your own validation Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits, targets, num_classes, eps=1e-6):
    """Soft multi-class Dice loss. logits: [B,C,H,W], targets: [B,H,W] (long)."""
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets_onehot, dims)
    cardinality = torch.sum(probs + targets_onehot, dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_per_class.mean()


def task_loss(logits, targets, num_classes, ce_weight=0.5, dice_weight=0.5):
    ce = F.cross_entropy(logits, targets)
    dl = dice_loss(logits, targets, num_classes)
    return ce_weight * ce + dice_weight * dl


def rewiring_loss(logits_clean, logits_corrupted, temperature=2.0):
    """KL(softmax(corrupted/T) || softmax(clean/T)), pooled spatially then averaged over batch."""
    logits_clean_pooled = logits_clean.flatten(2).mean(-1)
    logits_corrupted_pooled = logits_corrupted.flatten(2).mean(-1)
    log_p_corrupted = F.log_softmax(logits_corrupted_pooled / temperature, dim=-1)
    p_clean = F.softmax(logits_clean_pooled / temperature, dim=-1)
    return F.kl_div(log_p_corrupted, p_clean, reduction="batchmean")


def sparsity_loss(masks):
    """Fraction of active (>0.5) edges, averaged across a list of masks."""
    total, active = 0.0, 0.0
    for m in masks:
        total += m.numel()
        active += (m > 0.5).float().sum()
    return active / max(total, 1.0)


def diversity_loss(masks, margin=0.5):
    """Pairwise cosine-similarity hinge penalty across a list of masks."""
    if len(masks) < 2:
        device = masks[0].device if masks else "cpu"
        return torch.tensor(0.0, device=device)
    losses = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            mi = masks[i].flatten(1)
            mj = masks[j].flatten(1)
            sim = F.cosine_similarity(mi, mj, dim=-1)
            losses.append(torch.clamp(sim - margin, min=0.0))
    return torch.stack(losses).mean()


class NeuroMeshLoss(nn.Module):
    def __init__(self, num_classes, lambda_rewire=0.1, lambda_sparsity=0.01, lambda_diversity=0.001):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_rewire = lambda_rewire
        self.lambda_sparsity = lambda_sparsity
        self.lambda_diversity = lambda_diversity

    def forward(self, logits_clean, targets, logits_corrupted=None, masks=None):
        components = {"task": task_loss(logits_clean, targets, self.num_classes)}
        total = components["task"]

        if logits_corrupted is not None:
            components["rewire"] = rewiring_loss(logits_clean, logits_corrupted)
            total = total + self.lambda_rewire * components["rewire"]

        if masks is not None and len(masks) > 0:
            components["sparsity"] = sparsity_loss(masks)
            total = total + self.lambda_sparsity * components["sparsity"]

            components["diversity"] = diversity_loss(masks)
            total = total + self.lambda_diversity * components["diversity"]

        components["total"] = total
        return total, components
