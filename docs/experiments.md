# Experiments

Full frozen protocol: [`PROTOCOL_FREEZE_v1.md`](../PROTOCOL_FREEZE_v1.md) (repo root) — this document summarizes it.

## Design

- **n=30** real patients (Kaggle BraTS2020 `.h5` mirror, random subset, seed=42): 22 train / 4 val / 4 test, patient-level split.
- **96x96** resolution (compute accommodation — see `reproducibility.md`).
- **Four models**, identical protocol/hyperparameters/split: NeuroMesh, Plain UNet, DropoutUNet, StaticGatedUNet.
- **Five conditions** per model: clean, FLAIR-missing, T1-missing, T1ce-missing, T2-missing (single modality zeroed).
- **Metrics**: WT/TC/ET Dice, HD95, sensitivity, precision — full-patient-volume reconstruction (not per-slice averages), matching BraTS's own scoring convention.

## Validation vs. frozen test

- **Validation (n=4)**: used throughout development — checkpoint comparison, the epoch-4-vs-epoch-8 mechanistic check, all four models' baseline comparison. Safe to re-run.
- **Frozen test (n=4, disjoint patients)**: touched exactly once each for NeuroMesh and Plain UNet, per protocol. DropoutUNet and StaticGatedUNet have **not yet** been evaluated on test as of this snapshot — held back deliberately pending a finalized protocol for the full four-model comparison (see `results/manifests/RESULTS_MANIFEST.md`).

**Do not evaluate a new model or checkpoint against the frozen test set without updating `PROTOCOL_FREEZE_v1.md` first** — that document exists specifically so test-set use is a deliberate, recorded decision, not an incidental side effect of running a script.

## Timeline of what was actually run, in order

1. Pilot data/pipeline verification (single real patient, sanity-check only — see `figures/illustrative_synthetic/`).
2. NeuroMesh v1 trained, 4 epochs, 22 patients. Protocol frozen (`PROTOCOL_FREEZE_v1.md`) *before* touching test.
3. NeuroMesh v1 + Plain UNet evaluated on frozen test (once each).
4. Patient-level breakdown requested and produced from already-saved results (no new test-set touches).
5. Mechanism analysis (mask/gate) on NeuroMesh v1 — this pass used the *test* split (before a val/test toggle existed in the tooling) for diagnostic controller inspection only, after the official test evaluation above had already completed. Disclosed in `results/manifests/RESULTS_MANIFEST.md`.
6. NeuroMesh continued training to 8 epochs (same 22 patients, optimizer state reset at the boundary — see `reproducibility.md`). Full mechanism + WT/TC/ET re-evaluation, restricted to **val only** this time.
7. DropoutUNet and StaticGatedUNet trained (identical protocol) and evaluated on **val only**, producing the four-model comparison table.
8. Repository engineering (this codebase) — no new experiments; see `CHANGELOG.md`.

## What was explicitly *not* run in this pilot

Random partial-modality-dropout sweeps beyond single-modality-missing; correlated/cascading/Byzantine failure models (implemented in `neuromesh.analysis.failure_models`, tested, not yet run on real data); `EnsembleUNet` on real data; multi-seed variance estimation; any hyperparameter search; statistical significance testing (n=4 does not support it).

## Why n=4 twice, not once

Splitting 30 patients into 22/4/4 rather than a larger val or test fraction was a deliberate trade-off given the dataset-subset size available in this pilot's compute-constrained environment — not a recommendation for a production study. A follow-up with a full 369-patient pool (or the complete official BraTS2020/2021 release) would use substantially larger val/test splits.
