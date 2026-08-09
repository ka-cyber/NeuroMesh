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
import re
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

# Filename presets for the two BraTS-family layouts encountered so far.
# "adult" = BraTS 2020/2021 (CBICA / Kaggle mirrors): <case>_t1.nii[.gz], separator "_".
# "peds"  = BraTS-PEDs on TCIA (cancerimagingarchive.net/collection/brats-peds):
#           <case>-t1n.nii.gz, separator "-". Confirmed from the TCIA collection page's
#           documented naming convention (BraTS-PED-00XXX-000-t1n.nii.gz, etc.) as of
#           this writing -- verify against your own downloaded files, since TCIA can
#           revise conventions between dataset versions.
BRATS_PRESETS = {
    "adult": {
        "separator": "_",
        "modality_suffixes": {"t1": "t1", "t1ce": "t1ce", "t2": "t2", "flair": "flair"},
        "seg_suffix": "seg",
        # Official adult BraTS labels {0,1,2,4} (necrotic/non-enhancing core, edema,
        # enhancing tumor) -> remapped to contiguous {0,1,2,3}. Confirmed against the
        # official BraTS challenge label definition.
        "label_map": {4: 3},
        "num_classes": 4,  # background + NCR/NET + ED + ET
    },
    "peds": {
        "separator": "-",
        "modality_suffixes": {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"},
        "seg_suffix": "seg",
        # CONFIRMED against a real downloaded BraTS-PED-00001-000-seg.nii (see
        # `--inspect-seg`): raw labels are already a contiguous {0,1,2,3,4} --
        # background + 4 tumor subregions (ET/NET/CC/ED; BraTS-PEDs adds a Cystic
        # Component class that adult BraTS doesn't have) -- so no remap is needed.
        # This was previously left unverified in this codebase; it no longer is.
        "label_map": None,
        "num_classes": 5,  # background + NCR/NET + ED + CC + ET -- one more than adult
    },
}

# IMPORTANT: adult and pediatric BraTS use DIFFERENT numbers of output classes (4 vs 5)
# because BraTS-PEDs annotates a Cystic Component subregion that adult BraTS does not.
# A single model trained across both populations needs num_classes=5 throughout (adult
# cases simply never populate the Cystic Component class), NOT the adult preset's
# num_classes=4 -- that decision is left explicit here rather than made silently, since
# changing it changes the shape of every downstream loss/metric call. See README for the
# current status of unifying these into one label space.


class BraTSDataset(Dataset):
    """
    Loads real BraTS-family cases as 2D slices along a chosen axis. Configurable to
    match different BraTS-family naming conventions via `preset` ("adult" or "peds")
    or fully manual `separator` / `modality_suffixes` / `seg_suffix` / `label_map`
    arguments -- see BRATS_PRESETS above.

    Handles layout/extension quirks seen in practice, notably on the Kaggle mirror at
    kaggle.com/datasets/awsaf49/brats2020-training-data:

      * Nesting: some distributions put case folders directly under `root_dir`
        (official CBICA layout); others nest them one or two levels deeper,
        e.g. `root_dir/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/
        BraTS20_Training_001/`. This class searches recursively for any
        directory containing a segmentation file rather than assuming a fixed depth.
      * Extension: the official convention is `.nii.gz`, but multiple users
        of the awsaf49 Kaggle mirror report the files arriving as plain
        uncompressed `.nii` (same basename, no `.gz`). Both are accepted here.
    """

    def __init__(self, root_dir, slice_axis=2, modality_dropout_prob=0.0, target_size=(192, 192),
                 preset="adult", separator=None, modality_suffixes=None, seg_suffix=None, label_map=None):
        if not _HAS_NIBABEL:
            raise ImportError(
                "nibabel is required to load real BraTS volumes. "
                "Install with `pip install nibabel --break-system-packages`."
            )
        if preset is not None:
            if preset not in BRATS_PRESETS:
                raise ValueError(f"Unknown preset '{preset}'; choose from {list(BRATS_PRESETS)} or pass preset=None with manual args.")
            cfg = BRATS_PRESETS[preset]
            separator = cfg["separator"] if separator is None else separator
            modality_suffixes = cfg["modality_suffixes"] if modality_suffixes is None else modality_suffixes
            seg_suffix = cfg["seg_suffix"] if seg_suffix is None else seg_suffix
            label_map = cfg["label_map"] if label_map is None else label_map

        self.root_dir = root_dir
        self.slice_axis = slice_axis
        self.modality_dropout_prob = modality_dropout_prob
        self.target_size = target_size
        self.separator = separator
        self.modality_suffixes = modality_suffixes
        self.seg_suffix = seg_suffix
        self.label_map = label_map  # dict[int,int] or None (no remap -- raw label values passed through)

        self.case_dirs = self._discover_case_dirs(root_dir, self.separator, self.seg_suffix)
        if len(self.case_dirs) == 0:
            raise FileNotFoundError(
                f"No BraTS case directories found under {root_dir} (searched recursively for a "
                f"directory containing a *{self.separator}{self.seg_suffix}.nii or .nii.gz file, "
                f"preset='{preset}'). If this is the Kaggle awsaf49/brats2020-training-data mirror, "
                f"double check the extraction path -- it typically nests under "
                f"'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'. If this is BraTS-PEDs, "
                f"confirm preset='peds' was passed and the download actually extracted (Aspera "
                f"transfers can leave partial directories on interruption)."
            )

        sample_key = next(iter(self.modality_suffixes))
        sample_file = self._find_modality_file(self.case_dirs[0], sample_key)
        n_slices = nib.load(sample_file).get_fdata().shape[self.slice_axis]
        self.index = [(case_dir, s) for case_dir in self.case_dirs for s in range(n_slices)]
        self._cases = None  # set only by from_case_list()

    @classmethod
    def from_case_list(cls, cases, slice_axis=2, modality_dropout_prob=0.0, target_size=(192, 192), label_map=None):
        """
        Naming-convention-agnostic alternate constructor: build a dataset from an
        explicit list of cases, each a dict {'t1':path, 't1ce':path, 't2':path,
        'flair':path, 'seg':path}, instead of scanning a directory by filename pattern.

        Use this whenever files from different naming conventions/sources need to be
        paired for the same patient and no single preset's filename pattern covers
        both -- e.g. site-level raw exports (C{mapping_id}_{age}_{MODALITY}_to_SRI_
        defaced.nii, as distributed via CBTN) paired with officially-released BraTS-PEDs
        segmentation files (BraTS-PED-XXXXX-000-seg.nii.gz). See
        `discover_peds_site_raw_cases` for automated pairing of exactly that case via
        the official metadata crosswalk, rather than filename guessing.
        """
        instance = cls.__new__(cls)
        instance.root_dir = None
        instance.slice_axis = slice_axis
        instance.modality_dropout_prob = modality_dropout_prob
        instance.target_size = target_size
        instance.label_map = label_map
        instance.separator = None
        instance.modality_suffixes = None
        instance.seg_suffix = None
        instance._cases = cases

        sample_path = cases[0].get("flair") or cases[0].get("t1")
        n_slices = nib.load(sample_path).get_fdata().shape[slice_axis]
        instance.index = [(case, s) for case in cases for s in range(n_slices)]
        return instance

    @staticmethod
    def _discover_case_dirs(root_dir, separator, seg_suffix):
        """Recursively find every directory that contains a <sep><seg_suffix>.nii[.gz] file."""
        case_dirs = []
        suffix_gz = f"{separator}{seg_suffix}.nii.gz"
        suffix_plain = f"{separator}{seg_suffix}.nii"
        for dirpath, _, filenames in os.walk(root_dir):
            if any(f.endswith(suffix_gz) or f.endswith(suffix_plain) for f in filenames):
                case_dirs.append(dirpath)
        return sorted(case_dirs)

    def _find_modality_file(self, case_dir, modality_key):
        suffix = self.modality_suffixes[modality_key]
        for ext in ("nii.gz", "nii"):
            matches = glob.glob(os.path.join(case_dir, f"*{self.separator}{suffix}.{ext}"))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"No '{modality_key}' volume (suffix '{self.separator}{suffix}', .nii or .nii.gz) found in {case_dir}"
        )

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
        case_ref, slice_idx = self.index[idx]

        channels = []
        for modality in ["t1", "t1ce", "t2", "flair"]:
            if self._cases is not None:
                path = case_ref[modality]  # case_ref is a dict here (from_case_list mode)
            else:
                path = self._find_modality_file(case_ref, modality)  # case_ref is a directory path
            sl = self._center_crop_or_pad(self._load_slice(path, slice_idx))
            mean, std = sl.mean(), sl.std() + 1e-6
            channels.append((sl - mean) / std)  # per-slice z-score normalization

        x = torch.stack([torch.as_tensor(c, dtype=torch.float32) for c in channels], dim=0)

        if self._cases is not None:
            seg_path = case_ref["seg"]
        else:
            seg_path = glob.glob(os.path.join(case_ref, f"*{self.separator}{self.seg_suffix}.nii*"))[0]
        seg_sl = self._center_crop_or_pad(self._load_slice(seg_path, slice_idx)).copy()
        if self.label_map is not None:
            for src, dst in self.label_map.items():
                seg_sl[seg_sl == src] = dst
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


def build_peds_crosswalk(metadata_tsv_path):
    """
    Parses BraTS-PEDs_metadata.tsv (the official TCIA/CBTN metadata file) into a
    dict keyed by (MappingID, age_at_imaging_days) -> BraTS-SubjectID, plus a
    secondary dict keyed by MappingID alone -> BraTS-SubjectID for convenience when
    a filename only contains the mapping ID.

    This is the mechanism that resolves cases like
    'C1036890_4545_T1_to_SRI_defaced.nii' (site-level raw export, MappingID=C1036890,
    age=4545 days) to 'BraTS-PED-00001-000' (the official challenge subject ID used
    to name the released segmentation file) -- confirmed by cross-referencing both
    columns against a real downloaded metadata file, not inferred from the images.
    """
    by_mapping_and_age, by_mapping_only = {}, {}
    with open(metadata_tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            subject_id = row["BraTS-SubjectID"]
            mapping_id = row["MappingID"]
            age = row.get("Age at imaging (days)")
            if age:
                by_mapping_and_age[(mapping_id, str(age))] = subject_id
            by_mapping_only.setdefault(mapping_id, subject_id)
    return by_mapping_and_age, by_mapping_only


_PEDS_RAW_MODALITY_TOKENS = {"T1CE": "t1ce", "T1": "t1", "T2": "t2", "FL": "flair"}
_PEDS_RAW_FILENAME_RE = re.compile(
    r"^(?P<mapping_id>[A-Za-z0-9]+)_(?P<age>\d+)_(?P<modality>T1CE|T1|T2|FL)_to_SRI_defaced\.nii(\.gz)?$"
)


def discover_peds_site_raw_cases(raw_dir, metadata_tsv_path, seg_dir=None):
    """
    Auto-pairs site-level raw BraTS-PEDs exports (e.g. as distributed by CBTN:
    '<MappingID>_<age_days>_<MODALITY>_to_SRI_defaced.nii') with their officially-named
    segmentation files ('BraTS-PED-XXXXX-000-seg.nii[.gz]'), using the metadata TSV as
    the authoritative crosswalk -- NOT filename similarity or image-content heuristics.

    Args:
        raw_dir: directory containing the raw '<MappingID>_<age>_<MODALITY>_to_SRI_defaced.nii[.gz]' files.
        metadata_tsv_path: path to BraTS-PEDs_metadata.tsv.
        seg_dir: directory containing 'BraTS-PED-XXXXX-000-seg.nii[.gz]' files. Defaults to raw_dir.

    Returns:
        List of case dicts {'t1':path,'t1ce':path,'t2':path,'flair':path,'seg':path},
        suitable for BraTSDataset.from_case_list(). Cases missing any of the 4
        modalities or a matching segmentation file are skipped and reported, not
        silently dropped -- check the printed skip reasons.
    """
    seg_dir = seg_dir or raw_dir
    by_mapping_and_age, by_mapping_only = build_peds_crosswalk(metadata_tsv_path)

    groups = {}
    for fname in os.listdir(raw_dir):
        m = _PEDS_RAW_FILENAME_RE.match(fname)
        if not m:
            continue
        key = (m.group("mapping_id"), m.group("age"))
        modality = _PEDS_RAW_MODALITY_TOKENS[m.group("modality")]
        groups.setdefault(key, {})[modality] = os.path.join(raw_dir, fname)

    cases, skipped = [], []
    for (mapping_id, age), modality_paths in groups.items():
        missing_modalities = [m for m in ("t1", "t1ce", "t2", "flair") if m not in modality_paths]
        if missing_modalities:
            skipped.append((mapping_id, age, f"missing modalities: {missing_modalities}"))
            continue

        subject_id = by_mapping_and_age.get((mapping_id, age)) or by_mapping_only.get(mapping_id)
        if subject_id is None:
            skipped.append((mapping_id, age, "no crosswalk entry found in metadata TSV"))
            continue

        seg_matches = glob.glob(os.path.join(seg_dir, f"{subject_id}-seg.nii*"))
        if not seg_matches:
            skipped.append((mapping_id, age, f"resolved to {subject_id} but no matching -seg.nii[.gz] found in {seg_dir}"))
            continue

        case = dict(modality_paths)
        case["seg"] = seg_matches[0]
        case["subject_id"] = subject_id  # for logging/debugging; unused by BraTSDataset itself
        cases.append(case)

    if skipped:
        print(f"discover_peds_site_raw_cases: paired {len(cases)} case(s), skipped {len(skipped)}:")
        for mapping_id, age, reason in skipped:
            print(f"  {mapping_id}_{age}: {reason}")

    return cases


class H5SliceBraTSDataset(Dataset):
    """
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

    NOT YET VERIFIED BY US -- CONFIRM BEFORE TRUSTING CHANNEL SEMANTICS:
      - The exact modality order within 'image' (assumed [T1, T1ce, T2, FLAIR]
        here, matching the order used in the majority of public notebooks for
        this dataset, but we have not opened a real file from this specific
        dataset to confirm it ourselves). Run this file's --inspect mode on
        one real .h5 file after you download it:

            python data/brats_loader.py --inspect /path/to/volume_1_slice_75.h5

        and check that the FLAIR-like channel (index 3) is the brightest/most
        hyperintense one over CSF and edema before relying on modality-specific
        experiments (e.g. "drop the FLAIR channel").

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


def build_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Inspect real BraTS files before trusting any assumption this codebase makes about them."
    )
    p.add_argument("--inspect", type=str, help="Path to one .h5 slice file (H5SliceBraTSDataset format)")
    p.add_argument("--inspect-seg", type=str,
                   help="Path to one real *-seg.nii[.gz] file. Prints the actual integer label "
                        "values present, so you can confirm/build a label_map instead of trusting "
                        "an assumed one -- critical for BraTS-PEDs, whose label codes are not "
                        "documented on the TCIA collection page and are NOT assumed by this codebase.")
    args = p.parse_args()

    if args.inspect_seg:
        if not _HAS_NIBABEL:
            raise SystemExit("nibabel not installed. Run: pip install nibabel --break-system-packages")
        vol = nib.load(args.inspect_seg).get_fdata()
        unique, counts = np.unique(vol, return_counts=True)
        print(f"Unique integer labels in {args.inspect_seg}:")
        for u, c in zip(unique, counts):
            print(f"  label {u:.0f}: {int(c):,} voxels")
        print(
            "\nCompare this against the dataset's documented subregion names before setting "
            "label_map= on BraTSDataset -- do not assume it matches adult BraTS's {0,1,2,4} scheme."
        )

    if args.inspect:
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

    if not args.inspect and not args.inspect_seg:
        p.print_help()
