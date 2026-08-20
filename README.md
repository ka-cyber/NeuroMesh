# NeuroMesh

A topology-inspired bottleneck controller for multimodal brain-tumor MRI segmentation, studied here for behavior under **missing-modality conditions** (FLAIR/T1/T1ce/T2 dropout).

**Status: pilot study, n=4 validation patients, n=4 held-out test patients.** Nothing in this repository establishes clinical performance or generalization. Read that sentence again before citing any number below.

---

## What is NeuroMesh?

A U-Net for BraTS-style 4-modality segmentation, augmented with a bottleneck **controller**: a GRU hidden state + a graph-convolution component + a mask MLP that predicts a channel edge-activation mask, which gates the bottleneck's feature channels. It was designed to test the hypothesis that a model could learn to *rewire itself* in response to a missing or corrupted imaging modality.

## Scientific question

Does an adaptive, input-conditional bottleneck topology improve segmentation robustness to missing MRI modalities, relative to (a) a plain U-Net, (b) MC-dropout, and (c) a much simpler static learned channel gate?

## Current empirical finding — read this before anything else

Two things are true simultaneously, and the second complicates the first:

1. **NeuroMesh outperforms a plain U-Net on tumor-core (TC) and enhancing-tumor (ET) segmentation** in this pilot, fairly consistently across modality-missing conditions, and **underperforms on whole-tumor (WT) delineation specifically when FLAIR is missing** — a serious, reproducible robustness failure on this axis, not a minor one.

2. **We do not have evidence that this comes from adaptive topology rewiring.** Direct inspection of the controller's mask (`neuromesh analyze-controller`) shows it is close to input-*independent* — mean absolute difference between the mask under clean input and under any missing-modality condition is ~0.0000 across every patient we checked, at both 4 and 8 epochs of training. A much simpler control model (`StaticGatedUNet` — same bottleneck gating *operation*, no recurrence, no failure-signal conditioning, no adjacency mask, only 33K more parameters than plain) reproduces NeuroMesh's *specific* WT/FLAIR failure pattern almost exactly, collapsing on the same two patients. It does **not** fully reproduce NeuroMesh's TC advantage under T1ce loss, so "it's just the gate" isn't the whole story either — see [`docs/architecture.md`](docs/architecture.md) for the full breakdown.

**In short: the intended mechanism (dynamic, condition-dependent rewiring) is not clearly demonstrated. A largely static learned bottleneck transformation is a more defensible description of what this pilot's trained model actually does.** This distinction is the central finding of this repository, not a caveat buried in a limitations section.

## Key results (validation, n=4, pilot)

Mean WT / TC / ET Dice. Full per-patient breakdown, HD95, sensitivity/precision: [`results/validation/`](results/validation/), regenerable via `neuromesh compare`.

| condition | NeuroMesh (4ep) | Plain UNet | DropoutUNet | StaticGatedUNet |
|---|---|---|---|---|
| clean | 0.718 / 0.360 / 0.396 | 0.788 / 0.252 / 0.298 | 0.833 / 0.458 / 0.370 | 0.691 / 0.348 / 0.394 |
| FLAIR missing | **0.255** / 0.367 / 0.374 | 0.666 / 0.288 / 0.329 | 0.664 / 0.481 / 0.366 | **0.228** / 0.346 / 0.361 |
| T1 missing | 0.817 / 0.419 / 0.425 | 0.658 / 0.277 / 0.305 | 0.797 / 0.500 / 0.438 | 0.698 / 0.369 / 0.383 |
| T1ce missing | 0.616 / **0.000** / 0.000 | 0.738 / **0.000** / 0.000 | 0.820 / 0.447 / 0.000 | 0.659 / 0.378 / 0.001 |
| T2 missing | 0.624 / 0.359 / 0.395 | 0.751 / 0.239 / 0.294 | 0.829 / 0.471 / 0.387 | 0.764 / 0.337 / 0.389 |

ET under T1ce-missing is ~0.000 for **every model tested**, including the baselines — this looks like a genuine information-availability ceiling (T1ce is the only sequence showing gadolinium enhancement), not an architecture-specific weakness.

Frozen (n=4, touched **once**, per [`PROTOCOL_FREEZE_v1.md`](PROTOCOL_FREEZE_v1.md)): [`results/frozen/`](results/frozen/).

## Repository structure

```
neuromesh/            installable package (models, data, losses, training, evaluation, analysis)
scripts/              thin repo-level convenience wrappers
configs/              experiment configuration
results/
  frozen/             touched exactly once per model -- see PROTOCOL_FREEZE_v1.md
  validation/         development-time results, safe to have been regenerated
  mechanism/          controller mask/gate/topology analysis output
  manifests/          tables regenerated from the above (never hand-edited)
tests/
  unit/               component-level tests
  regression/          tests for specific bugs found and fixed (see below)
  integration/         end-to-end synthetic-data pipeline tests
docs/                 architecture, dataset, experiments, reproducibility, troubleshooting
paper/                manuscript drafts (still contain unfilled TODOs -- see paper/README)
figures/              reproducible figure-generation scripts
artifacts/checkpoints/  trained model weights (gitignored; see artifacts/checkpoints/README.md)
```

## Installation

```bash
git clone <this-repo>
cd NeuroMesh
pip install -e ".[all]"     # or ".[data]" for just nibabel/h5py, ".[dev]" for pytest/ruff
```

Requires Python >= 3.10. CPU-only works (this pilot's own experiments were run on 1 CPU core, 4GB RAM, no GPU — see [`docs/reproducibility.md`](docs/reproducibility.md)); a GPU is not required but is recommended for anything beyond a tiny pilot.

## Dataset preparation

This repository does **not** ship any patient data. See [`docs/dataset.md`](docs/dataset.md) for exact sourcing instructions (the pre-sliced Kaggle BraTS2020 `.h5` mirror, or the official CBICA NIfTI release) and the patient-level split protocol used.

## Quick start (no real data required)

```bash
python train.py --epochs 3          # synthetic MockBraTSDataset smoke test -- verifies the
                                     # pipeline runs end to end, produces NO clinically meaningful numbers
pytest tests/                       # full test suite, synthetic fixtures only, no download needed
```

## Training (real data)

```bash
neuromesh train --dataset brats_h5 --data-root <path> --metadata-csv <path> \
    --model neuromesh --base-ch 16 --image-size 96 --batch-size 8 --epochs 4 \
    --val-fraction 0.15 --test-fraction 0.15 --split-seed 42 --run-name my_run
```

Supports `--model {neuromesh, plain, dropout, ensemble, static_gate}`. Checkpoint/resume built in (`--max-seconds` for time-limited environments) — see [`docs/reproducibility.md`](docs/reproducibility.md) for exactly how the pilot's own training was chunked across a 1-core sandbox with a hard per-call time limit.

## Evaluation

```bash
neuromesh evaluate --checkpoint <ckpt> --split val --results-out <out.json>   # safe, repeatable
neuromesh evaluate --checkpoint <ckpt> --split test --results-out <out.json>  # touches frozen test -- see PROTOCOL_FREEZE_v1.md first
```

Reports per-patient WT/TC/ET Dice, HD95, sensitivity, precision — full-volume reconstruction, not per-slice averages (matching how BraTS itself scores submissions).

## Missing-modality evaluation

Built into `neuromesh evaluate` — every run reports clean + all four single-modality-missing conditions. Modality channel identity (`neuromesh.data.MODALITY_ORDER`) was determined empirically from the actual downloaded data, not assumed; see [`docs/dataset.md`](docs/dataset.md) for the verification method and its confidence level.

## Baselines

`PlainUNet`, `DropoutUNet` (MC-dropout), `EnsembleUNet`, `StaticGatedUNet` (the mechanistic control described above). `EnsembleUNet` is implemented and tested but not yet trained/evaluated on real data in this pilot.

## Mechanistic analysis

```bash
neuromesh analyze-controller --checkpoint <ckpt> --out-dir <dir> --split val --condition clean
neuromesh analyze-controller --checkpoint <ckpt> --out-dir <dir> --split val   # representative-slice snapshots
```

Extracts the controller's failure-signal, gate, and mask statistics per condition — the tooling behind the negative finding described above. Never run against the frozen test set except where explicitly documented as part of the original protocol (see `results/mechanism/epoch4_TOUCHED_TEST_SET/README.md`).

## Reproducing the reported results

```bash
neuromesh reproduce        # prints the exact commands used, in order
neuromesh compare --results-dir . --out-dir results/manifests   # regenerates every table in this README from stored JSON
```

## Reproducibility

Every experiment's exact hyperparameters, split seed, patient IDs, checkpoint SHA-256, and hardware are recorded in [`PROTOCOL_FREEZE_v1.md`](PROTOCOL_FREEZE_v1.md) and [`docs/reproducibility.md`](docs/reproducibility.md). This pilot was run on 1 CPU core / 4GB RAM / no GPU — expect training to be slow at any meaningful scale; this is a disclosed compute limitation, not a design choice.

## Testing

```bash
pytest tests/                              # everything, synthetic fixtures only
pytest tests/regression/                   # bug-specific regressions (see below)
```

**A real bug found and fixed during real-data evaluation**: the controller's recurrent hidden state is carried across batches. Real data's final batch of an epoch is rarely the same size as the rest; the synthetic defaults (32 samples / batch 4) always divided evenly and never triggered this. Fixed by resetting the hidden state on batch-size change — regression-tested in `tests/regression/test_hidden_state_batch_mismatch.py`.

## Docker

```bash
docker build -t neuromesh .
```

(Not verified to build in the environment this repository was assembled in — no Docker daemon was available. Review before relying on it.)

## Citation

See [`CITATION.cff`](CITATION.cff). No manuscript DOI exists yet — do not cite one.

## License

MIT — see [`LICENSE`](LICENSE).

## Limitations

- n=4 validation, n=4 test. This is a pilot, not a generalization study.
- 96×96 downsampled resolution (native BraTS is 240×240) — a compute accommodation for a 1-CPU-core/4GB-RAM environment, not a methodological choice.
- Modality channel identity was determined by our own visual/statistical analysis of the downloaded data, not confirmed against the dataset publisher's documentation.
- Dynamic topology adaptation, the architecture's original motivation, was not clearly observed at either 4 or 8 epochs of training.
- `EnsembleUNet` and the full correlated/cascading/Byzantine failure-model matrix are implemented but not yet run on real data.
- No statistical significance testing has been performed (n=4 does not support it) — Dice differences are reported descriptively, not as significant/non-significant.
- Single random seed for the reported training runs.

## Contact

Open an issue on this repository.
