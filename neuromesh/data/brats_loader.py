"""
data/brats_loader.py

Two data sources:

  1. BraTSDataset -- loads real BraTS-format cases (T1, T1ce, T2, FLAIR + seg
     label, each a .nii.gz volume) from a directory tree matching the layout
     used by the BraTS 2020/2021 challenges:

         <root>/<case_id>/<case_id>_t1.nii.gz
         <root>/<case_id>/<case_id>_t1ce.nii.gz
         <root>/<case_id>/<case_id>_t2.nii.gz
         <root>/<case_id>/<case_id>_flair.nii.gz
         <root>/<case_id>/<case_id>_seg.nii.gz

     This requires `pip install nibabel --break-system-packages`, and requires
     that YOU have already obtained real BraTS volumes under the appropriate
     data-use agreement (e.g. the UPenn CBICA portal, or one of the Kaggle
     mirrors). Nothing here fetches, includes, or references any actual scan
     data -- it only knows how to read files once they exist on your disk.

  2. MockBraTSDataset -- a synthetic, dependency-free generator that produces
     4-channel tensors and paired segmentation masks with plausible shapes and
     a randomly placed "lesion" blob, so the rest of the pipeline (models/,
     utils/, train.py) can be exercised end-to-end with zero external data.
     Outputs of this class are NOT real MRI data and results computed on it
     say nothing about real clinical performance -- it exists purely to let
     you smoke-test the code.
"""

import os
import csv
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
    _HAS_NIBABEL = True
except ImportError:
    _HAS_NIBABEL = False

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


BRATS_MODALITIES = ["t1", "t1ce", "t2", "flair"]


class BraTSDataset(Dataset):
    """
    Loads real BraTS cases as 2D slices along a chosen axis.

    Handles two layout/extension quirks seen in practice, notably on the
    Kaggle mirror at kaggle.com/datasets/awsaf49/brats2020-training-data:

      * Nesting: some distributions put case folders directly under `root_dir`
        (official CBICA layout); others nest them one or two levels deeper,
        e.g. `root_dir/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
        BraTS20_Training_001/`. This class searches recursively for any
        directory containing a `*_seg.nii*` file rather than assuming a fixed
        depth.
      * Extension: the official convention is `.nii.gz`, but multiple users
        of the awsaf49 Kaggle mirror report the files arriving as plain
        uncompressed `.nii` (same basename, no `.gz`). Both are accepted here.
    """

    def __init__(self, root_dir, slice_axis=2, modality_dropout_prob=0.0, target_size=(192, 192),
                 case_dirs=None):
        """
        Args:
            root_dir: passed through even when `case_dirs` is given, purely for
                error messages -- discovery is skipped if `case_dirs` is set.
            case_dirs: optional explicit list of case directories to use. Pass
                this (rather than relying on auto-discovery of `root_dir`) when
                you need a patient-level train/val/test split -- see
                `split_nifti_case_dirs` below. Auto-discovering per-split root
                directories independently is *not* equivalent to this, since a
                single top-level root passed three times would just discover
                the same full case list three times.
        """
        if not _HAS_NIBABEL:
            raise ImportError(
                "nibabel is required to load real BraTS volumes. "
                "Install with `pip install nibabel --break-system-packages`."
            )
        self.root_dir = root_dir
        self.slice_axis = slice_axis
        self.modality_dropout_prob = modality_dropout_prob
        self.target_size = target_size

        self.case_dirs = case_dirs if case_dirs is not None else self._discover_case_dirs(root_dir)
        if len(self.case_dirs) == 0:
            raise FileNotFoundError(
                f"No BraTS case directories found under {root_dir} "
                f"(searched recursively for a directory containing a *_seg.nii or *_seg.nii.gz file). "
                f"If this is the Kaggle awsaf49/brats2020-training-data mirror, double check the "
                f"extraction path -- it typically nests under "
                f"'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'."
            )

        sample_flair = self._find_modality_file(self.case_dirs[0], "flair")
        n_slices = nib.load(sample_flair).get_fdata().shape[self.slice_axis]
        self.index = [(case_dir, s) for case_dir in self.case_dirs for s in range(n_slices)]

    @staticmethod
    def _discover_case_dirs(root_dir):
        """Recursively find every directory that contains a *_seg.nii[.gz] file."""
        case_dirs = []
        for dirpath, _, filenames in os.walk(root_dir):
            if any(f.endswith("_seg.nii.gz") or f.endswith("_seg.nii") for f in filenames):
                case_dirs.append(dirpath)
        return sorted(case_dirs)

    @staticmethod
    def _find_modality_file(case_dir, modality):
        # Prefer .nii.gz (official convention) but fall back to plain .nii
        # (observed on the Kaggle awsaf49 mirror).
        for ext in ("nii.gz", "nii"):
            matches = glob.glob(os.path.join(case_dir, f"*_{modality}.{ext}"))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No '{modality}' volume (.nii or .nii.gz) found in {case_dir}")

    def _load_slice(self, path, slice_idx):
        vol = nib.load(path).get_fdata()
        return vol.take(indices=slice_idx, axis=self.slice_axis)

    def _center_crop_or_pad(self, arr):
        h, w = arr.shape
        th, tw = self.target_size
        if h > th:
            top = (h - th) // 2
            arr = arr[top: top + th, :]
        if w > tw:
            left = (w - tw) // 2
            arr = arr[:, left: left + tw]
        h, w = arr.shape
        pad_h, pad_w = max(th - h, 0), max(tw - w, 0)
        if pad_h or pad_w:
            arr = torch.nn.functional.pad(
                torch.from_numpy(arr).float(), (0, pad_w, 0, pad_h)
            ).numpy()
        return arr

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case_dir, slice_idx = self.index[idx]

        channels = []
        for modality in BRATS_MODALITIES:
            path = self._find_modality_file(case_dir, modality)
            sl = self._center_crop_or_pad(self._load_slice(path, slice_idx))
            mean, std = sl.mean(), sl.std() + 1e-6
            channels.append((sl - mean) / std)  # per-slice z-score normalization

        x = torch.stack([torch.as_tensor(c, dtype=torch.float32) for c in channels], dim=0)

        seg_path = self._find_modality_file(case_dir, "seg")
        seg_sl = self._center_crop_or_pad(self._load_slice(seg_path, slice_idx)).copy()
        seg_sl[seg_sl == 4] = 3  # BraTS labels {0,1,2,4} -> contiguous {0,1,2,3}
        y = torch.as_tensor(seg_sl, dtype=torch.long)

        if self.modality_dropout_prob > 0:
            for c in range(x.size(0)):
                if random.random() < self.modality_dropout_prob:
                    x[c] = 0.0

        return x, y


class MockBraTSDataset(Dataset):
    """
    Synthetic stand-in for BraTSDataset. Generates random 4-channel "MRI-like"
    slices with a randomly placed elliptical "lesion" region so the rest of the
    pipeline can be unit-tested with zero external dependencies and zero real
    data.

    THIS IS NOT REAL DATA and must never be described as one in a manuscript.
    """

    def __init__(self, n_samples=64, image_size=(128, 128), num_classes=4, seed=0):
        self.n_samples = n_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self._seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        h, w = self.image_size
        rng = random.Random(self._seed * 100000 + idx)  # deterministic per-index
        torch_gen = torch.Generator().manual_seed(self._seed * 100000 + idx)

        x = torch.randn(4, h, w, generator=torch_gen) * 0.3
        x[0] += 0.2   # T1-like baseline contrast
        x[1] += 0.3   # T1ce-like
        x[2] += 0.1   # T2-like
        x[3] += 0.4   # FLAIR-like

        y = torch.zeros(h, w, dtype=torch.long)
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")

        n_blobs = rng.randint(1, 2)
        for _ in range(n_blobs):
            cy, cx = rng.randint(h // 4, 3 * h // 4), rng.randint(w // 4, 3 * w // 4)
            ry, rx = rng.randint(5, max(6, h // 6)), rng.randint(5, max(6, w // 6))
            blob = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2 <= 1.0

            label = rng.randint(1, self.num_classes - 1)
            y[blob] = label
            # Purely illustrative intensity bumps -- not a real MRI physics model.
            x[2][blob] += 0.6
            x[3][blob] += 0.9
            x[1][blob] += 0.4 if label == self.num_classes - 1 else 0.1

        return x, y


class H5SliceBraTSDataset(Dataset):
    """
    MODALITY_ORDER (module-level constant just below this class) is the single
    source of truth for image-channel identity -- import it rather than
    hard-coding channel indices/names elsewhere. (A prior version of this
    codebase duplicated the mapping across train_real.py and eval_checkpoint.py
    and only updated one copy when the mapping was corrected -- exactly the
    bug this constant exists to prevent.)

    Loads the PRE-SLICED BraTS2020 .h5 format distributed as
    kaggle.com/datasets/awsaf49/brats2020-training-data, indexed via its
    accompanying `BraTS20_Training_Metadata.csv`.

    CONFIRMED (from the Kaggle dataset description and multiple independent
    published notebooks that load these exact files):
      - metadata CSV columns: slice_path, target, volume, slice,
        label0_pxl_cnt, label1_pxl_cnt, label2_pxl_cnt, background_ratio
      - 369 volumes (patients) x 155 slices = 57,195 total slices, matching
        the official BraTS2020 training case count and standard (240,240,155)
        volume geometry
      - each .h5 file has exactly two datasets: 'image' with shape [240,240,4]
        and 'mask' with shape [240,240,3]
      - 'mask' stores THREE INDEPENDENT BINARY CHANNELS (one per tumor
        sub-region: BraTS labels {1: necrotic/non-enhancing core, 2: edema,
        4: enhancing tumor}), not a single integer class map like the raw
        NIfTI '_seg' files. This loader converts that into a single integer
        map {0=background, 1, 2, 3} for compatibility with NeuroMeshUNet /
        NeuroMeshLoss, assuming per-pixel channel exclusivity (true for the
        standard BraTS label decomposition; verify on your own data if you
        see unexpectedly low tumor-class counts).

    IMAGE CHANNEL ORDER -- CHECKED AGAINST REAL DOWNLOADED DATA (not assumed):
      Three independent checks on real downloaded .h5 files, all converging on
      the same answer, which CONTRADICTS this loader's original assumption
      ([T1, T1ce, T2, FLAIR] at indices [0,1,2,3]):

        1. Region-contrast test: mean intensity inside each of the mask's 3
           tumor sub-regions vs. outside, across 9-10 real tumor-containing
           slices. Channel 2 showed a large, specific jump only in the region
           that would be "enhancing tumor" under the standard BraTS mask
           ordering [necrotic, edema, enhancing] -- the textbook signature of
           gadolinium contrast (T1ce).
        2. Direct visual inspection of ventricle/CSF appearance (the classic
           T1-vs-T2-vs-FLAIR discriminator) on tight crops across 2
           independent real slices from different patients: channel 3 has
           bright/hyperintense CSF (only T2 does this); channels 0, 1, 2 all
           have dark/suppressed CSF.
        3. Same crops also show a bright enhancing streak specifically at the
           choroid plexus (inside the ventricle) on channel 2 only --
           choroid plexus lacks a blood-brain barrier and is a well-known
           structure that enhances with gadolinium contrast, again pointing
           to channel 2 = T1ce specifically (and corroborating check #1
           independently).

      Combining these: channel 0 = FLAIR, channel 1 = T1, channel 2 = T1ce,
      channel 3 = T2. I.e. the real order is [FLAIR, T1, T1ce, T2], not
      [T1, T1ce, T2, FLAIR].

      Confidence: reasonably high (three independent signals agree, including
      a specific anatomical landmark for T1ce), but this is still our own
      visual/statistical read on a sample of slices, not a documented fact
      from the dataset publisher -- if this experiment reaches a manuscript,
      have someone with radiology background sanity-check a rendered slice
      before stating it as fact, and cite that it was independently verified
      rather than assumed.

    LICENSING: BraTS data requires citing Menze et al. (2015, IEEE TMI) and
    the relevant Bakas et al. papers regardless of which mirror it's pulled
    from -- carry that citation into the manuscript's data section.

    Args:
        metadata_csv: path to BraTS20_Training_Metadata.csv
        data_root: local directory containing the actual .h5 files. The CSV's
            `slice_path` column is a Kaggle-notebook-relative path
            (".../content/data/volume_41_slice_0.h5") that won't exist
            locally -- only the basename is stable, so it's re-joined with
            `data_root`.
        volume_ids: optional iterable of int volume IDs to restrict to (use
            with build_volume_split for leakage-free train/val/test splits).
        tumor_only: if True, keep only slices with target==1 (visible tumor).
        min_tumor_pixels: drop slices whose summed tumor pixel count (across
            all 3 label channels, per the CSV columns) is below this.
        modality_dropout_prob: independently zero each of the 4 channels with
            this probability at __getitem__ time.
    """

    def __init__(self, metadata_csv, data_root, volume_ids=None, tumor_only=False,
                 min_tumor_pixels=0, modality_dropout_prob=0.0,
                 image_key="image", mask_key="mask"):
        if not _HAS_H5PY:
            raise ImportError("h5py is required. Install with `pip install h5py --break-system-packages`.")

        self.data_root = data_root
        self.modality_dropout_prob = modality_dropout_prob
        self.image_key = image_key
        self.mask_key = mask_key

        rows = []
        with open(metadata_csv, newline="") as f:
            for row in csv.DictReader(f):
                if volume_ids is not None and int(row["volume"]) not in volume_ids:
                    continue
                if tumor_only and row["target"] != "1":
                    continue
                total_tumor_px = (
                    int(row["label0_pxl_cnt"]) + int(row["label1_pxl_cnt"]) + int(row["label2_pxl_cnt"])
                )
                if total_tumor_px < min_tumor_pixels:
                    continue
                rows.append(row)

        if not rows:
            raise ValueError(
                "No rows matched the given filters in the metadata CSV. "
                "Check volume_ids / tumor_only / min_tumor_pixels."
            )
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, slice_path):
        # Only the basename is portable across environments (the CSV's
        # directory prefix reflects the original Kaggle notebook's mount path).
        return os.path.join(self.data_root, os.path.basename(slice_path))

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = self._resolve_path(row["slice_path"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Confirm `data_root` points at the local directory "
                f"holding the extracted .h5 files (basename expected: {os.path.basename(row['slice_path'])})."
            )

        with h5py.File(path, "r") as hf:
            image = np.array(hf[self.image_key], dtype=np.float32)  # [H, W, 4]
            mask = np.array(hf[self.mask_key], dtype=np.float32)    # [H, W, 3]

        x = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # [4, H, W]
        for c in range(x.size(0)):
            mean, std = x[c].mean(), x[c].std() + 1e-6
            x[c] = (x[c] - mean) / std

        mask_t = torch.from_numpy(mask)  # [H, W, 3]
        has_tumor = mask_t.sum(dim=-1) > 0
        tumor_class = mask_t.argmax(dim=-1) + 1  # -> {1,2,3}
        y = torch.zeros(mask_t.shape[0], mask_t.shape[1], dtype=torch.long)
        y[has_tumor] = tumor_class[has_tumor]

        if self.modality_dropout_prob > 0:
            for c in range(x.size(0)):
                if random.random() < self.modality_dropout_prob:
                    x[c] = 0.0

        return x, y


# Single source of truth for image-channel identity -- see the
# H5SliceBraTSDataset docstring above for how this was verified. Import this
# rather than re-hard-coding channel indices/names in scripts that use it.
MODALITY_ORDER = ["FLAIR", "T1", "T1ce", "T2"]  # index -> modality name


def build_volume_split(metadata_csv, val_fraction=0.15, test_fraction=0.15, seed=42):
    """
    Split BraTS *volumes* (patients) -- not individual slices -- into
    train/val/test sets. Splitting by slice instead of by patient leaks
    information (adjacent slices of the same tumor are highly correlated) and
    will inflate validation/test metrics; this is a common and important bug
    to avoid in a manuscript that reviewers will look for specifically.

    Returns three sets of int volume IDs: (train_ids, val_ids, test_ids).
    """
    with open(metadata_csv, newline="") as f:
        volumes = sorted(set(int(row["volume"]) for row in csv.DictReader(f)))

    rng = random.Random(seed)
    rng.shuffle(volumes)
    n = len(volumes)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))

    val_ids = set(volumes[:n_val])
    test_ids = set(volumes[n_val:n_val + n_test])
    train_ids = set(volumes[n_val + n_test:])
    return train_ids, val_ids, test_ids


def split_nifti_case_dirs(root_dir, val_fraction=0.15, test_fraction=0.15, seed=42):
    """
    Patient-level train/val/test split for the raw-NIfTI BraTS layout, mirroring
    `build_volume_split` (which does the same thing for the pre-sliced .h5
    mirror via its metadata CSV). Splits *case directories* (patients), not
    slices, for the same leakage reason documented on `build_volume_split`.

    Returns three lists of case-directory paths, suitable for passing directly
    as `BraTSDataset(root_dir, case_dirs=...)`.
    """
    case_dirs = BraTSDataset._discover_case_dirs(root_dir)
    if len(case_dirs) == 0:
        raise FileNotFoundError(f"No BraTS case directories found under {root_dir}")

    rng = random.Random(seed)
    rng.shuffle(case_dirs)
    n = len(case_dirs)
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))

    val_dirs = case_dirs[:n_val]
    test_dirs = case_dirs[n_val:n_val + n_test]
    train_dirs = case_dirs[n_val + n_test:]
    return train_dirs, val_dirs, test_dirs


def build_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect a real BraTS .h5 slice file to confirm keys/shapes/value ranges "
                    "before trusting any channel-order assumptions made in H5SliceBraTSDataset."
    )
    p.add_argument("--inspect", type=str, required=True, help="Path to one .h5 slice file")
    args = p.parse_args()

    if not _HAS_H5PY:
        raise SystemExit("h5py not installed. Run: pip install h5py --break-system-packages")

    with h5py.File(args.inspect, "r") as hf:
        print(f"keys in {args.inspect}: {list(hf.keys())}")
        for k in hf.keys():
            arr = np.array(hf[k])
            print(f"  '{k}': shape={arr.shape} dtype={arr.dtype} min={arr.min():.4f} max={arr.max():.4f}")
        if "image" in hf.keys():
            img = np.array(hf["image"])
            if img.ndim == 3 and img.shape[-1] == 4:
                print("\nPer-channel mean intensity (helps confirm modality order -- FLAIR is")
                print("typically the most hyperintense over edema/CSF relative to T1):")
                for c in range(4):
                    print(f"  channel {c}: mean={img[..., c].mean():.4f} max={img[..., c].max():.4f}")
