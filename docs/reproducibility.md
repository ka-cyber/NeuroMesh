# Reproducibility

## Hardware the reported experiments actually ran on

1 CPU core, 4GB RAM, no GPU. This is disclosed because it directly explains two design choices below that would otherwise look unmotivated:

- **96x96 training resolution** (native BraTS is 240x240). A compute accommodation, not a methodological claim. `--image-size` in `neuromesh train` controls this; re-running at native resolution on real compute is the natural first step before treating any pilot number as final.
- **Checkpoint/resume training** (`--max-seconds` in `neuromesh train`). The execution environment this pilot ran in imposed a hard per-call wall-clock limit (~280-380s) and did not preserve background processes across calls. Training therefore proceeds in chunks: each invocation trains until `--max-seconds` elapses, saves model + optimizer state + step count, and exits cleanly (code 0); re-running the identical command resumes. This is real infrastructure, not a workaround specific to one bad run — see `neuromesh/training/train_real.py`.

## Every result's provenance

Every entry in `results/validation/` and `results/frozen/` is machine-readable JSON recording: the full CLI args used, dataset split sizes, patient IDs, device, wall-clock time, timestamp, and (where available) checkpoint SHA-256. Nothing was hand-typed into a table without a JSON file behind it — see `results/manifests/RESULTS_MANIFEST.md`.

## What "continuing training" actually means here (epoch 4 -> epoch 8)

`neuromesh_e8.pt` was produced via `--init-from-checkpoint checkpoints/neuromesh_real30.pt`, which loads model weights only. **Optimizer state (AdamW momentum/variance) was not preserved across this boundary** — the original run's final checkpoint didn't save it. This is a disclosed, minor deviation from perfectly continuous training, not full resumption. If exact continuity matters for a future run, save optimizer state in the *final* checkpoint too (currently only the in-progress/resumable checkpoint does).

## Determinism

`torch.manual_seed(args.seed)` is set at the start of each training run (default `--seed 0`). DataLoader shuffling order was **not** additionally seeded per-resume-chunk during this pilot's multi-call chunked training (13 resumed CLI calls for the original 4-epoch run) — so exact batch order is not bit-reproducible across a from-scratch re-run, though results should be statistically similar given the same seed governs weight initialization. Full bitwise reproducibility was not a design goal of this pilot; if needed for a future release, seed the DataLoader's generator explicitly per resume and persist its RNG state in the checkpoint.

## What is and isn't tested by CI

`tests/` runs entirely on synthetic fixtures (`MockBraTSDataset`) — no real dataset download is required or should ever be required for CI (see `.github/workflows/tests.yml`). This means CI verifies code correctness (shapes, gradients, loss finiteness, the hidden-state regression, metric correctness on synthetic labels) but does **not** verify the real-data scientific results in `results/` — those can only be verified by re-running against real BraTS data, which is a manual, licensed-data-dependent step (`neuromesh reproduce`).

## Software environment

See `pyproject.toml` for exact dependency bounds. The reported experiments used PyTorch 2.13 (CPU build), Python 3.12, scipy 1.17, on Ubuntu (container). No GPU/CUDA toolkit was present or used.
