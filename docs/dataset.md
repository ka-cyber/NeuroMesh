# Dataset

No patient data is shipped in this repository. Obtain BraTS through an official channel and comply with its data-use terms; cite Menze et al. (2015, IEEE TMI) and the relevant Bakas et al. papers regardless of mirror.

## Source used for this pilot

Kaggle `awsaf49/brats2020-training-data` — a pre-sliced `.h5` mirror of BraTS2020. 369 patients x 155 slices each. This pilot used a **random 30-patient subset** (seed=42, unconditioned on tumor size — every one of the 369 full-set volumes has 14.8-65.8% tumor-visible slices, so this carries no known selection bias).

```bash
pip install kaggle   # requires a free Kaggle account + API token, see Kaggle's docs
kaggle datasets download -d awsaf49/brats2020-training-data
```

Each `.h5` file has two datasets: `image` (`[240, 240, 4]`, float) and `mask` (`[240, 240, 3]`, binary sub-region channels). The loader (`neuromesh.data.H5SliceBraTSDataset`) converts the 3-channel mask into a single integer label map `{0=background, 1=necrotic/non-enhancing core, 2=edema, 3=enhancing tumor}`.

## Modality channel order — verified, not assumed

The `.h5` format does not document which of the 4 image channels is which MRI sequence. An earlier version of this codebase assumed `[T1, T1ce, T2, FLAIR]` without verification. That assumption was checked against real downloaded data and found to be **wrong**.

`neuromesh.data.MODALITY_ORDER = ["FLAIR", "T1", "T1ce", "T2"]` — determined via three independent checks, all converging:

1. **Region-contrast test**: mean intensity inside each of the mask's 3 tumor sub-regions vs. outside, across real tumor slices. Channel 2 shows a large, specific jump only in the region matching "enhancing tumor" under the standard BraTS mask ordering — the textbook gadolinium-contrast signature (T1ce).
2. **CSF/ventricle appearance**: direct visual inspection across independent slices from different patients. Channel 3 has bright/hyperintense CSF (only T2 does this among the four sequences); channels 0, 1, 2 all have dark/suppressed CSF.
3. **Choroid plexus enhancement**: the same crops show a bright enhancing streak specifically at the choroid plexus (inside the ventricle) on channel 2 only — a well-known contrast-enhancement landmark, independently corroborating check #1's identification of channel 2 as T1ce.

**Confidence**: reasonably high — three independent signals agree, including a specific anatomical landmark. This is still our own visual/statistical read on a sample of slices, not documentation from the dataset publisher. If this matters for a specific venue, have someone with radiology background verify a rendered slice (`neuromesh.data.brats_loader` has an `--inspect` mode for exactly this) before stating it as fact in a manuscript.

Full detail and the exact verification code: see the `H5SliceBraTSDataset` docstring in `neuromesh/data/brats_loader.py`.

## Patient-level split (frozen for v1)

30-patient subset -> `build_volume_split(metadata_csv, val_fraction=0.15, test_fraction=0.15, seed=42)`:

- **Train (n=22)**: 13, 14, 16, 17, 45, 58, 72, 102, 113, 120, 141, 215, 217, 230, 280, 288, 303, 309, 328, 347, 360, 367
- **Val (n=4)**: 112, 126, 259, 333
- **Test (n=4)**: 48, 53, 115, 302 — untouched except per `PROTOCOL_FREEZE_v1.md`

Splitting is patient-level (not slice-level) specifically to prevent slice leakage between splits — all 155 slices of a given patient stay in the same split.

## Alternative: raw NIfTI

`neuromesh.data.BraTSDataset` supports the official CBICA-portal NIfTI layout directly (case directories containing `*_t1.nii(.gz)`, `*_t1ce.nii(.gz)`, `*_t2.nii(.gz)`, `*_flair.nii(.gz)`, `*_seg.nii(.gz)`), with `split_nifti_case_dirs()` providing the same patient-level split logic. Not the format used for this pilot's reported results, but fully supported and tested.
