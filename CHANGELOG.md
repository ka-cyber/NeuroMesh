# Changelog

## [0.1.0] -- pre-publication research snapshot

Initial reproducibility release corresponding to the pilot study described in
`docs/experiments.md`.

### Added
- Real-data training/evaluation pipeline (`neuromesh train`, `neuromesh evaluate`), replacing the prior synthetic-only skeleton as the authoritative implementation.
- `StaticGatedUNet` mechanistic control model.
- Full-volume WT/TC/ET/HD95/sensitivity/precision evaluation (`neuromesh.evaluation.metrics.compute_patient_region_metrics`), replacing per-slice-averaged raw-class Dice as the primary reported metric.
- Controller mechanism analysis tooling (`neuromesh analyze-controller`): mask sensitivity, gate selectivity, topology visualization.
- Checkpoint/resume training support for time-limited compute environments.
- Empirically-verified `MODALITY_ORDER`, replacing an unverified assumption in the prior skeleton.
- Regression test for a real hidden-state/batch-size-mismatch bug found during real-data evaluation.
- Frozen experimental protocol (`PROTOCOL_FREEZE_v1.md`) and results provenance manifest (`results/manifests/`).
- Installable package structure, CLI, this documentation set.

### Fixed
- Hidden-state batch-size mismatch in `evaluate_modality_dropout` and the training loop (see README and `tests/regression/`).
- Modality-channel-order assumption, corrected from an unverified `[T1,T1ce,T2,FLAIR]` guess to an empirically-verified `[FLAIR,T1,T1ce,T2]`.

### Scientific findings recorded in this release
- NeuroMesh outperforms Plain UNet on TC/ET in this pilot; underperforms on WT specifically under FLAIR loss.
- Controller mask does not show measurable input-dependent structure at 4 or 8 epochs of training on this pilot's 22-patient training set.
- A parameter-light static-gating control reproduces NeuroMesh's WT/FLAIR failure mode but not its TC/T1ce-loss behavior.
- ET is undetectable under T1ce loss for all four models tested -- consistent with an information-availability ceiling, not an architecture-specific failure.

### Known limitations (see README for full list)
- n=4/n=4 val/test -- pilot scope only, not a generalization study.
- 96x96 resolution (compute accommodation).
- Single seed, no significance testing.
- `EnsembleUNet` and the full failure-model matrix implemented but not yet run on real data.
