"""
utils/failure_models.py

The supplementary material describes four failure models (Section C.1-C.4:
node, edge, cascading, Byzantine) and a generalization-transfer experiment
across them (Table 6) -- but no injection code for the correlated, cascading,
or Byzantine cases exists anywhere in this project; only independent random
per-channel modality dropout is implemented in utils/evaluation.py. This file
implements the other three so the generalization-transfer experiment the
draft describes is actually possible to run for real.

These operate on a feature tensor at any chosen layer (e.g. the U-Net
bottleneck) via a forward hook, rather than only on the 4 raw input channels
-- which is necessary because "20% of network edges/channels" in the original
draft's claims refers to internal features, not just the 4 MRI sequences.

All four functions have the same signature: (features, rate, generator) ->
corrupted_features, so they're interchangeable in a training or eval loop.
"""

import torch


def random_failure(features: torch.Tensor, rate: float, generator=None) -> torch.Tensor:
    """Each channel independently zeroed with probability `rate`. (Baseline case.)"""
    B, C = features.shape[0], features.shape[1]
    keep = (torch.rand(B, C, generator=generator) > rate).float().to(features.device)
    shape = [B, C] + [1] * (features.dim() - 2)
    return features * keep.view(*shape)


def correlated_failure(features: torch.Tensor, rate: float, generator=None,
                        neighbor_boost: float = 0.3) -> torch.Tensor:
    """
    Spatially/index-correlated failures: if channel i fails, channel i+1 has an
    elevated (not independent) probability of also failing. Mirrors the
    supplementary's stated empirical assumption ("if neuron (l,i) fails,
    probability neighbor (l,i+-1) also fails increases by 30%") applied along
    the channel index as the nearest available notion of "adjacency" for a
    generic feature vector.
    """
    B, C = features.shape[0], features.shape[1]
    device = features.device
    base_fail = (torch.rand(B, C, generator=generator).to(device) < rate)

    fail = base_fail.clone()
    # propagate to immediate neighbors with boosted probability
    boosted_rate = min(rate + neighbor_boost, 1.0)
    neighbor_roll = torch.rand(B, C, generator=generator).to(device)
    left_failed = torch.roll(base_fail, shifts=1, dims=1)
    right_failed = torch.roll(base_fail, shifts=-1, dims=1)
    adjacent_to_failure = left_failed | right_failed
    fail = fail | (adjacent_to_failure & (neighbor_roll < boosted_rate))

    keep = (~fail).float()
    shape = [B, C] + [1] * (features.dim() - 2)
    return features * keep.view(*shape)


def cascading_failure(features: torch.Tensor, rate: float, generator=None,
                       n_steps: int = 4, step_growth: float = 1.6) -> torch.Tensor:
    """
    Temporally-growing failure: simulates a fault that starts small and
    spreads over `n_steps` (e.g. successive inference calls / a thermal event
    worsening), each step failing more channels than the last. Returns the
    features as they would appear at the FINAL step; use
    cascading_failure_sequence() if you need every intermediate step (e.g. to
    train/evaluate a recurrent controller's response over time).
    """
    seq = cascading_failure_sequence(features, rate, generator, n_steps, step_growth)
    return seq[-1]


def cascading_failure_sequence(features: torch.Tensor, rate: float, generator=None,
                                n_steps: int = 4, step_growth: float = 1.6):
    """Returns a list of length n_steps, each an increasingly-corrupted copy of `features`."""
    B, C = features.shape[0], features.shape[1]
    device = features.device
    shape = [B, C] + [1] * (features.dim() - 2)

    cumulative_fail = torch.zeros(B, C, dtype=torch.bool, device=device)
    out = []
    for step in range(n_steps):
        step_rate = min(rate * (step_growth ** step), 1.0)
        new_fail = torch.rand(B, C, generator=generator).to(device) < step_rate
        cumulative_fail = cumulative_fail | new_fail
        keep = (~cumulative_fail).float()
        out.append(features * keep.view(*shape))
    return out


def byzantine_failure(features: torch.Tensor, rate: float, generator=None,
                       polarity_flip: float = -10.0) -> torch.Tensor:
    """
    Adversarial/corrupted-value failure: rather than zeroing, selected
    channels are replaced with a large-magnitude polarity-flipped version of
    themselves (worst-case-from-the-network's-perspective corruption, as
    opposed to silent zeroing). This is a materially different, harder
    condition than the other three -- a controller that learns to zero out
    low-mask-value channels handles silent failures but must additionally
    learn to DETECT anomalous magnitude/sign to handle this case, which is
    exactly the limitation the supplementary itself calls out ("Byzantine
    attacks: adversarially crafted node outputs can still fool the
    controller"). Don't expect the same robustness numbers here as for the
    other three failure types without an explicit outlier-detection mechanism.
    """
    B, C = features.shape[0], features.shape[1]
    device = features.device
    corrupt = torch.rand(B, C, generator=generator).to(device) < rate
    shape = [B, C] + [1] * (features.dim() - 2)
    corrupt_mask = corrupt.float().view(*shape)
    corrupted_features = features * polarity_flip
    return features * (1 - corrupt_mask) + corrupted_features * corrupt_mask


FAILURE_MODELS = {
    "random": random_failure,
    "correlated": correlated_failure,
    "cascading": cascading_failure,
    "byzantine": byzantine_failure,
}


def evaluate_transfer_matrix(model, dataloader, num_classes, device, rate=0.2,
                              max_batches=None, feature_hook_module=None):
    """
    Reproduces the STRUCTURE of the supplementary's Table 6 (train-on-one,
    test-on-another transfer matrix) as real, runnable code. This function
    only measures transfer for a model already trained under one of the four
    conditions -- it does not itself claim any particular transfer accuracy.
    You must actually train four separate model instances (one per failure
    type used during training) and pass each one through this function
    against all four test-time failure types to populate a real matrix; nothing
    here should be filled in with the original draft's 90-95% figures without
    doing that.

    Applies the corruption at the model's bottleneck feature map via a forward
    hook (default: model.controller, matching NeuroMeshUNet's bottleneck
    controller) rather than at the raw input, matching "network edges/channels"
    as used in the original failure-model description.
    """
    import torch.nn.functional as F
    from utils.evaluation import dice_per_class

    hook_target = feature_hook_module or model.controller
    results = {}

    for name, fn in FAILURE_MODELS.items():
        handle = None

        def make_hook(failure_fn):
            def hook(module, inputs, output):
                # NeuroMeshLayer returns (y, h_new, mask); corrupt y.
                y, h_new, mask = output
                y_corrupted = failure_fn(y, rate)
                return (y_corrupted, h_new, mask)
            return hook

        handle = hook_target.register_forward_hook(make_hook(fn))
        model.eval()
        all_dices = []
        with torch.no_grad():
            for i, (x, y_true) in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                x, y_true = x.to(device), y_true.to(device)
                logits, _, _ = model(x)
                all_dices.append(dice_per_class(logits, y_true, num_classes))
        handle.remove()

        mean_per_class = torch.tensor(all_dices).mean(dim=0).tolist()
        results[name] = {
            "mean_dice_per_class": mean_per_class,
            "mean_dice": sum(mean_per_class) / len(mean_per_class),
        }

    return results
