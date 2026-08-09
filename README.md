# NeuroMesh

A self-rewiring graph-neural topology controller for fault-tolerant medical
image segmentation. This repo implements a GCN+GRU controller that predicts a
dynamic edge-activation mask at the bottleneck of a 4-channel (T1/T1ce/T2/FLAIR)
U-Net, trained with a composite task + rewiring + sparsity + diversity loss.

## Status (read this before citing any number from this repo)

**Engineering: complete and tested.** Every module below has been implemented,
run, and checked — 26/26 automated tests pass, the full pipeline runs
end-to-end on synthetic data, it has been verified against one real
BraTS-format case, and the manuscript compiles cleanly.

**Science: not yet done.** No version of this model has been trained on real
BraTS data. Every number currently in `results/` comes from either synthetic
placeholder data or a model trained only on that synthetic data — they confirm
the *pipeline* works, not that the *architecture* segments real tumors well.
The one real-data test we ran (see `results/`) returned near-zero tumor-class
Dice, which is the expected and correct outcome for a model that has never
seen a real brain, not a finding about NeuroMesh's real-world capability
either way. See `paper/tmi/neuromesh.tex or paper/media/main.tex` for a full accounting — every
placeholder is marked with a visible `\TODO{}`.

**Do not report any Dice/latency/robustness number from this repo as a
validated result until it has been produced by training on real, properly
split BraTS data.**

## What's real and what's a baseline you get from open-source

Our loss/metric implementations were cross-checked against
[MONAI](https://github.com/Project-MONAI/MONAI) (agreement within ~0.001-0.0013;
see `results/`), and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) is a
reasonable open-source reference point to benchmark against once real
training happens — neither is vendored in this repo, just noted here as
recommended companions.

## Repository structure

```
data/brats_loader.py        Real BraTS loaders (raw NIfTI + pre-sliced HDF5
                             Kaggle mirror) + a synthetic MockBraTSDataset for
                             dependency-free pipeline testing.
models/layers.py            GraphConvLayer (normalized GCN) + NeuroMeshLayer
                             (the MLP->GRU->GCN->mask-MLP controller).
models/segmentation.py      NeuroMeshUNet: 4-channel U-Net with the controller
                             gating the bottleneck.
models/baselines.py         PlainUNet / DropoutUNet (MC-Dropout) / EnsembleUNet
                             -- real comparison baselines (not in the original
                             draft this project started from).
utils/loss.py                Composite loss (task Dice+CE, rewiring KL,
                             sparsity, diversity).
utils/evaluation.py          Dice-under-modality-dropout + measured latency.
utils/failure_models.py      Random / correlated / cascading / Byzantine
                             fault injection on bottleneck features, plus a
                             transfer-matrix evaluator.
train.py                     End-to-end runnable script (synthetic data by
                             default -- swap in real BraTS via brats_loader.py).
figures/                     Scripts that generate every figure, plus their
                             current (synthetic/pipeline-check) output PNGs.
tests/                       26 automated tests (pytest).
paper/tmi/neuromesh.tex        IEEE T-MI submission (IEEEtran), compiles cleanly.
paper/media/main.tex           Medical Image Analysis submission (Elsevier
                                elsarticle), compiles cleanly, real author
                                block, CRediT/competing-interest/data-availability
                                sections included. Self-contained Overleaf
                                project: main.tex + references.bib + figures/.
                                Every unfilled result marked with \TODO{}.
results/                     Current run outputs (see Status above).
Dockerfile, requirements.txt, .github/workflows/tests.yml
                             Reproducible environment + CI.
```

## Setup

```bash
pip install -r requirements.txt
```

Or with Docker (CPU only — see the Dockerfile header for GPU instructions):

```bash
docker build -t neuromesh .
docker run neuromesh
```

## Running things

```bash
# Full automated test suite
pytest tests/ -v

# Train + evaluate on synthetic data (works with zero external data/deps)
python train.py --epochs 5 --n-train 32 --n-val 16

# Regenerate any figure
python figures/fig_a_cortical_analogy.py
python figures/fig_b_dropout_panel.py
python figures/fig_c_latency_benchmark.py --budget-ms 100

# Inspect a real downloaded BraTS .h5 slice file's actual keys/shapes
python data/brats_loader.py --inspect /path/to/volume_1_slice_75.h5
```

## Using real data — adult and pediatric BraTS, one codebase

`data/brats_loader.py` handles three real-data sources plus the synthetic one,
none of which ship with any actual data — download data yourself under the
relevant data-use agreement and point the loader at it locally.
`.gitignore` already excludes `data/raw/`, `*.nii(.gz)`, and `*.h5` so you
won't accidentally commit patient data.

- **`BraTSDataset(root_dir, preset="adult")`** — adult BraTS 2020/2021, official
  NIfTI layout (`<case>_t1.nii[.gz]`, etc.). Handles both the flat CBICA layout
  and the nested Kaggle mirror layout, and both `.nii`/`.nii.gz`. 4 output
  classes (background + NCR/NET + ED + ET), raw labels `{0,1,2,4}` remapped to
  contiguous `{0,1,2,3}`.
- **`BraTSDataset(root_dir, preset="peds")`** — the official TCIA BraTS-PEDs
  release naming (`-t1n.nii.gz`, `-t1c.nii.gz`, `-t2w.nii.gz`, `-t2f.nii.gz`,
  `-seg.nii.gz`). 5 output classes (background + NCR/NET + ED + **Cystic
  Component** + ET — CC is a pediatric-only subregion adult BraTS doesn't
  have). Raw labels are already contiguous `{0,1,2,3,4}`, confirmed against a
  real downloaded case with `--inspect-seg` — no remap needed.
- **`BraTSDataset.from_case_list(cases)`** + **`discover_peds_site_raw_cases()`**
  — for real-world data that arrives with *inconsistent* naming across files,
  which is common with site-level exports. BraTS-PEDs as distributed by CBTN
  ships segmentation files under the official challenge ID
  (`BraTS-PED-00001-000-seg.nii.gz`) but raw modality volumes under a
  completely different site convention (`C1036890_4545_T1_to_SRI_defaced.nii`
  — `MappingID_age-in-days_MODALITY`). No filename pattern bridges those two,
  so `discover_peds_site_raw_cases()` instead cross-references both against
  `BraTS-PEDs_metadata.tsv` (the official crosswalk between MappingID+age and
  BraTS-SubjectID) to pair them correctly — verified against a real case, not
  inferred from image content. Use `BraTSDataset.from_case_list()` directly if
  you already have your own case-to-file mapping from any other source.
- **`H5SliceBraTSDataset`** — the pre-sliced adult `.h5` Kaggle mirror
  (`kaggle.com/datasets/awsaf49/brats2020-training-data`), indexed via its
  `BraTS20_Training_Metadata.csv`. Includes `build_volume_split()` for
  **patient-level** train/val/test splitting — critical for BraTS, since
  splitting by slice instead of by patient leaks information between splits.

**Training one model across both populations:** adult and pediatric BraTS
don't share a class count (4 vs 5) because of the Cystic Component subregion.
Nothing in this codebase silently unifies them — if you want a single model
across both, decide explicitly whether to (a) train two separate output heads,
or (b) adopt `num_classes=5` everywhere and let adult cases simply never
populate the Cystic Component class (adult raw labels `{0,1,2,4}` already sit
inside `{0,1,2,3,4}` without remapping if you skip the adult preset's `4→3`
step). Neither is implemented as a default; both are viable, and the choice
affects every downstream loss/metric call, so it's left as an explicit
decision rather than an assumption.

## What's still needed before this is a submittable result

See the Discussion/Limitations section of `paper/tmi/neuromesh.tex or paper/media/main.tex` for the full
list. Short version: a GPU, real BraTS training with a real patient-level
split, hyperparameter search, baseline comparisons (code for these already
exists in `models/baselines.py`), and — if this is ever positioned as a
clinical tool rather than a research architecture — IRB approval, clinical
validation, and a regulatory pathway. None of that is shortcut-able by more
code.

## Citation

This project uses the BraTS dataset. If you use it, cite:

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark
  (BraTS)," IEEE TMI, 2015.
- Bakas et al., "Advancing the Cancer Genome Atlas glioma MRI collections with
  expert segmentation labels and radiomic features," Scientific Data, 2017.
- Bakas et al., "Identifying the best machine learning algorithms for brain
  tumor segmentation...," arXiv:1811.02629, 2019.

## License

MIT (see `LICENSE`) — chosen as a permissive default since none was specified.
Confirm with your institution before publishing; see the note in `LICENSE`.
