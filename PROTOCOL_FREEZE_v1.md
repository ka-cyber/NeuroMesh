# Protocol freeze: NeuroMesh pilot, real-data v1

Written and finalized **before** running any evaluation on the test split.
Nothing below may change based on test-set results. If NeuroMesh underperforms
on test, that's the recorded result for this protocol version -- any fix
becomes v2, with its own fresh validation cycle and its own untouched test
evaluation. Test-set numbers do not feed back into this document.

## 1. Data

- Source: Kaggle `awsaf49/brats2020-training-data` (pre-sliced `.h5` mirror
  of BraTS2020), NOT the full 369-patient set -- a 30-patient random subset
  (seed=42, drawn from the full pool, unconditioned on tumor size/content;
  see prior session notes -- every one of the 369 volumes has 14.8-65.8%
  tumor-visible slices, so this random draw carries no known selection bias).
- 30 volumes x 155 slices = 4,650 total slices.

## 2. Patient-level split (frozen)

Computed by `build_volume_split(metadata_csv, val_fraction=0.15,
test_fraction=0.15, seed=42)` over the 30-volume subset's metadata CSV.
This function shuffles volume IDs with `random.Random(42)` and slices the
list -- deterministic and reproducible from the CSV alone.

- **Train (n=22)**: `[13, 14, 16, 17, 45, 58, 72, 102, 113, 120, 141, 215,
  217, 230, 280, 288, 303, 309, 328, 347, 360, 367]`
- **Val (n=4)**: `[112, 126, 259, 333]` -- used for model selection / the
  earlier checkpoint-correction sanity check. Not used again below.
- **Test (n=4)**: `[48, 53, 115, 302]` -- **UNTOUCHED as of this freeze**.
  Not seen during training, validation monitoring, architecture choice,
  hyperparameter selection, or the modality-order verification (that used
  patients 45, 48, 141 -- wait, see Note below).

**Note on patient 48 and 141**: the modality-channel-order verification
(visual CSF/choroid-plexus check, done before this freeze) used slices from
volumes 45, 48, and 141. Volume 48 is in the test set; volumes 45 and 141 are
in train. This is a **minor, acknowledged leak**: no label information or
model-relevant signal was used from patient 48 (only raw image channel
appearance, to determine modality identity, which is a fixed property of the
dataset/scanner protocol, not of any specific patient's pathology) -- but
strictly, zero test-patient pixels should have been looked at before this
freeze, and one was. Recorded here rather than silently ignored. If this
matters for a venue with strict data-leakage requirements, the modality-order
check should be redone on a train-only patient before a v2 freeze.

## 3. Modality channel order (frozen, verified empirically)

`data.brats_loader.MODALITY_ORDER = ["FLAIR", "T1", "T1ce", "T2"]`

Verified via: (1) tumor-subregion-specific contrast test, (2) CSF/ventricle
brightness check, (3) choroid-plexus contrast-enhancement landmark -- see
`H5SliceBraTSDataset` docstring in `data/brats_loader.py` for full detail.
Confidence: reasonably high, three independent signals agree, but this is
our own visual/statistical read, not documentation from the dataset
publisher. Flag this provenance in the manuscript rather than stating it as
established fact.

## 4. Preprocessing

- Per-channel z-score normalization (mean 0, std 1), computed per-slice, per
  the existing `H5SliceBraTSDataset.__getitem__`.
- Mask conversion: BraTS's 3 binary sub-region channels -> single integer map
  {0=background, 1=necrotic/non-enhancing core, 2=edema, 3=enhancing tumor},
  assuming per-pixel channel exclusivity (standard for this decomposition).
- **Downsampled to 96x96** (native is 240x240) via bilinear interpolation
  (image) / nearest-neighbor (mask), through `train_real.py`'s
  `_ResizeWrapper`. This is a compute-budget accommodation for this sandbox's
  1-CPU-core/4GB-RAM hardware, not a methodological choice -- flagged
  explicitly in the manuscript as a limitation, to be re-run at native
  resolution on real compute before any number here is treated as final.
- No spatial augmentation beyond the existing per-channel modality dropout
  (training-time only, p=0.15 per channel, independent).

## 5. Model configuration (frozen)

- Architecture: `NeuroMeshUNet(in_channels=4, num_classes=4, base_ch=16)`
  -- 19,149,108 parameters.
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-5.
- Loss: `NeuroMeshLoss` (task Dice+CE + rewiring KL + sparsity + diversity),
  default weights (unmodified from repo defaults -- not tuned for this run).
- Batch size: 8.
- Epochs: 4 (1,708 total optimizer steps = 4 x 427 batches/epoch).
- Training-time modality dropout: p=0.15 per channel, independent.
- Seed: `--seed 0` (torch.manual_seed), though note DataLoader shuffling
  order was not additionally seeded per-resume-chunk (training ran across 13
  resumed CLI calls due to sandbox time limits -- see checkpoint below).

## 6. Checkpoint (frozen)

- File: `checkpoints/neuromesh_real30.pt`
- SHA-256: `19108deeffbca2723a4edf9dc046164712b5b69decbe98d7bf884448af152962`
- Trained: 1,708/1,708 steps complete, 1,950s cumulative wall-clock (CPU).
- **This exact checkpoint, and no other, is what gets evaluated on test.**
  If anything about the model changes (more epochs, different base_ch,
  different loss weights, bug fixes to the architecture itself), that
  produces a NEW checkpoint requiring a NEW protocol freeze -- it does not
  get evaluated against this test set under this protocol version.

## 7. Metrics (frozen definitions)

Computed per-patient, on the FULL reconstructed volume (155 slices stacked
in native z-order), not averaged per-slice -- matching how BraTS itself
scores submissions:

- **WT** (whole tumor) = labels {1, 2, 3}
- **TC** (tumor core) = labels {1, 3} (excludes edema)
- **ET** (enhancing tumor) = label {3} only
- **Dice**: standard, eps=1e-6 smoothed.
- **HD95**: 95th-percentile symmetric surface distance via boundary erosion +
  Euclidean distance transform (`scipy.ndimage`). Assumes 1mm isotropic
  voxel spacing (BraTS's standard preprocessing) -- NOT independently
  re-verified from this specific Kaggle mirror's files (no NIfTI affine
  metadata available in the `.h5` format to check directly). Reported as
  `None` (not 0, not inf) when either mask is empty -- undefined, not a
  fabricated number.
- **Sensitivity / precision**: per-region, `None` when the ground-truth
  region is entirely absent (0/0 undefined).
- Predictions reconstructed at training resolution (96x96) then
  nearest-neighbor-upsampled to native 240x240 before computing metrics
  against the native-resolution ground truth mask.

## 8. Test-set evaluation plan (frozen, in order)

1. **Clean test**: WT/TC/ET Dice, HD95, sensitivity, precision. Per-patient
   and mean across the 4 test patients.
2. **Single-modality-missing** (4 conditions): FLAIR missing, T1 missing,
   T1ce missing, T2 missing -- one modality zeroed at a time, same metrics.

**Explicitly NOT run in this pass** (per protocol -- establish clean +
single-modality performance first before justifying the larger study):
random partial-dropout sweeps, correlated/cascading/Byzantine failure
models, baseline comparisons (Dropout/Ensemble/Plain UNet), multi-seed
variance, ablations, mechanistic controller analysis.

## 9. Rule

Test-set numbers are recorded as-is. No re-training, no hyperparameter
changes, no architecture changes in response to test results under this
protocol version. Any change starts a v2 freeze.
