# Architecture

## Data flow

```
Input: 4 MRI modalities [FLAIR, T1, T1ce, T2] (channel order verified empirically -- see dataset.md)
        |
Preprocessing: per-slice per-channel z-score normalization, optional modality dropout (training only)
        |
U-Net encoder (4 downsampling stages, DoubleConv blocks)
        |
Bottleneck [B, base_ch*16, h, w]
        |
NeuroMesh controller (see below)
        |
U-Net decoder (4 upsampling stages, skip connections)
        |
1x1 conv -> per-pixel class logits {background, necrotic/non-enhancing core, edema, enhancing tumor}
```

## The controller: intended mechanism

`neuromesh.models.NeuroMeshLayer` (`neuromesh/models/components.py`) was designed as:

1. Global-average-pool the bottleneck feature map to a per-channel vector.
2. Auto-derive a `failure_signal`: which bottleneck channels are numerically dead (`abs(x) < 1e-6`), as a proxy for "something upstream has failed or gone missing."
3. Feed `(pooled_features, failure_signal)` through a GRU cell, carrying a hidden state across calls, to produce a `rewired` vector and an updated hidden state.
4. A GCN-style component combines the GRU output with a learned adjacency matrix to predict a `mask` — an edge-activation matrix over the bottleneck's channels (`[B, C, C]`, C = 256 at `base_ch=16`).
5. `gate = sigmoid(rewired)`, applied channel-wise to the bottleneck: `x5 = x5 * gate`.

**The hypothesis this was built to test**: that the mask and gate would learn to reconfigure themselves conditionally, based on which modality (if any) is missing — i.e., genuine input-dependent topology adaptation, not just a fixed learned transform.

## The controller: observed mechanism (this pilot)

Using `neuromesh analyze-controller`, we measured the controller's actual behavior on the same 4 val/test patients across 5 conditions (clean + 4 single-modality-missing), at 4 and 8 epochs of training. See `results/mechanism/` for the raw data this section summarizes.

### The mask is not measurably input-dependent

For every patient x condition pair checked, at both epoch 4 and epoch 8:

- Mean absolute difference between the clean-condition mask and the failure-condition mask: **~0.00000** (smaller at epoch 8 than epoch 4, not larger).
- Fraction of the 65,536 mask entries (256x256) that change by more than 0.05 between clean and any failure condition: **0.0000**, for every patient checked.
- Visual inspection (rendered mask heatmaps) shows an unstructured, near-random pattern, visually indistinguishable across all 5 conditions.

This directly contradicts the "dynamic rewiring" framing as an established fact for this trained model. Whatever the mask MLP has learned, it is not conditioning meaningfully on which modality is present or absent.

### The gate changed substantially with more training, but not conditionally

Between epoch 4 and epoch 8 (same 22 training patients, continued training, no architecture change):

- Mean gate value dropped from ~0.45 to ~0.365.
- Fraction of the 256 channels gated below 0.1 (strongly suppressed) rose from ~4% to ~40%.
- This shift is nearly identical whether the input is clean or has a modality missing — the gate became a much more selective *fixed* filter with training, not a more *conditional* one.

One partial exception: under T1ce-missing specifically, the fraction of strongly-suppressed channels dropped further (to ~0.7% at epoch 4) than under any other condition, consistently across all 4 patients. This is a real, reproducible, condition-specific signal — but a small one, and it corresponds to the controller becoming *less* selective (not more, and not in an obviously compensatory way) exactly where TC/ET performance collapses to zero.

### A simple static control reproduces much of the behavior

`StaticGatedUNet` has the identical bottleneck gating *operation* as NeuroMeshUNet, but the gate comes from a plain feedforward MLP on the pooled bottleneck vector — no recurrence, no failure-signal input, no adjacency mask. It is **not parameter-matched** to the full NeuroMeshLayer (19.1M params vs. 1.98M) — most of that gap is the mask MLP's `[C,C]` output layer, which is exactly the component this control excludes by design.

Result: StaticGatedUNet collapses on the *same two patients*, under the *same condition* (FLAIR-missing, WT specifically), as NeuroMesh does, almost exactly. This is consistent with NeuroMesh's WT/FLAIR fragility coming from "a learned bottleneck gate exists at all," not from anything specific to the controller's recurrent/graph machinery.

It does **not**, however, reproduce NeuroMesh's TC recovery under T1ce-missing (StaticGate: 0.378, NeuroMesh: 0.000, Plain: 0.000, Dropout: 0.447) — the pattern here groups Dropout with StaticGate, not with NeuroMesh, and doesn't split cleanly along "has a bottleneck gate" at all. We do not have an explanation for this and are not offering a post-hoc one.

## What we can and cannot claim

**Can claim**: in this pilot, NeuroMesh's TC/ET advantage over plain U-Net is real and fairly consistent; its WT/FLAIR fragility is real, reproducible, and closely mirrored by a much simpler static-gating control; the controller's mask has not developed measurable input-dependent structure at the training scale used here (22 patients, up to 8 epochs, 96x96).

**Cannot claim**: that NeuroMesh performs dynamic topology adaptation; that its advantages come specifically from the recurrent/graph controller machinery rather than the mere presence of bottleneck gating; that any of this generalizes beyond 4 validation and 4 test patients; that longer training would or wouldn't eventually produce condition-dependent mask structure (only 4 and 8 epochs were checked at the time of writing).

## Open questions this motivates

- Does mask differentiation ever emerge with substantially more training, or is 65,536 free parameters (a fully unconstrained mask over 256 channels) simply too large a search space for 22 training patients to constrain, regardless of epoch count? (Raised, not resolved, in `PROTOCOL_FREEZE_v1.md` follow-up discussion.)
- What explains the TC/T1ce-missing pattern that groups {Dropout, StaticGate} against {Plain, NeuroMesh}, which cuts against the simplest "it's just the gate" story?
- Would a smaller, more constrained mask (e.g., a low-rank or block-structured adjacency) learn condition-dependent structure faster than the current fully free 256x256 formulation?
