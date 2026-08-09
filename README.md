# NeuroMesh

**NeuroMesh** is a self-reconfiguring graph-neural topology controller for
fault-tolerant multimodal medical image segmentation.

The framework integrates a **GCN–GRU topology controller** into the bottleneck
of a four-channel U-Net to dynamically predict an edge-activation mask over
feature representations. The controller is trained using a composite objective
combining segmentation performance with topology-rewiring, sparsity, and
diversity regularization.

NeuroMesh is designed to investigate whether **dynamic computational topology
reconfiguration** can provide increased robustness to modality degradation and
internal feature faults in multimodal medical image segmentation.

---

## Project status

### Engineering and implementation — complete

The current release contains a complete, executable implementation of the
NeuroMesh framework.

- 26/26 automated tests pass.
- The complete training and evaluation pipeline runs end-to-end on synthetic
  data.
- Real adult BraTS2021 data have been successfully loaded, processed, and
  evaluated.
- Real pediatric BraTS-PEDs data have been successfully loaded, processed, and
  evaluated.
- Adult and pediatric modality naming conventions are handled by the same
  loader framework.
- Real segmentation labels have been used for single-case verification.
- Synthetic modality-dropout and feature-fault experiments are implemented.
- Latency benchmarks are implemented and measured.
- Baseline architectures are included.
- The project includes reproducible environment configuration through
  `requirements.txt`, Docker, and continuous integration.
- Figure-generation scripts are included with the repository.

### Scientific validation — deliberately scoped

The current release should be understood as a **methodological and
implementation-validation release**, not as a population-level clinical or
benchmark study.

NeuroMesh has been exercised on real BraTS data, including:

- one adult BraTS2021 multimodal case;
- one pediatric BraTS-PEDs multimodal case;
- corresponding real segmentation labels for quantitative pipeline
  verification.

These real-data experiments demonstrate that the complete pipeline can ingest
real multimodal MRI data, perform preprocessing and inference, and compare
predictions against real segmentation annotations.

However, the current model checkpoint used for these real-case verification
experiments was trained on synthetic data. Therefore, the single-case real-data
experiments **must not be interpreted as estimates of real-world segmentation
performance or generalization**.

In particular, this repository does **not** currently claim:

- state-of-the-art BraTS segmentation performance;
- population-level Dice or HD95 performance;
- superiority over nnU-Net or other established segmentation systems;
- clinical diagnostic performance;
- clinical utility;
- generalization across independent patient cohorts.

The purpose of the current release is to provide a transparent, reproducible
implementation and validation framework from which systematic real-data
training and benchmarking can be performed.

---

## What has been validated

The current NeuroMesh implementation has been evaluated at several levels.

### 1. Software and unit-level validation

The repository contains automated tests covering the major model, loss,
evaluation, and data-processing components.

```
```
26 / 26 automated tests passing
2. Synthetic end-to-end training

A dependency-light synthetic dataset is provided to verify that the complete
pipeline can execute without requiring external medical-imaging data.

The synthetic pipeline exercises:

data generation
      ↓
multimodal input
      ↓
U-Net encoder
      ↓
NeuroMesh topology controller
      ↓
dynamic edge mask
      ↓
U-Net decoder
      ↓
segmentation
      ↓
loss + evaluation

Synthetic results are used to verify implementation behavior and should not be
interpreted as clinical or real-world segmentation results.

3. Real adult BraTS2021 verification

The framework has been exercised using a real adult BraTS2021 case containing
the four standard MRI modalities:

T1
T1ce
T2
FLAIR

The corresponding real segmentation annotation was also used for
single-case verification.

This verifies the complete path from real multimodal MRI input through
preprocessing, model inference, and comparison with a real segmentation label.

4. Real pediatric BraTS-PEDs verification

The framework has also been exercised using a real pediatric BraTS-PEDs case
with:

T1n
T1c
T2w
T2f

The corresponding pediatric segmentation annotation is handled using the
pediatric BraTS label convention.

This verifies that the same software framework can accommodate the distinct
modality naming and label conventions used by adult and pediatric BraTS
datasets.

5. Modality-dropout evaluation

Controlled modality-dropout experiments are implemented to investigate the
behavior of the segmentation pipeline when one or more input modalities become
unavailable.

The current synthetic experiments demonstrate that the evaluation machinery
can simulate modality degradation and quantify its effect.

Real-data population-level modality-dropout benchmarking remains future work.

6. Feature-fault evaluation

NeuroMesh includes controlled feature-level fault models covering:

random faults;
correlated faults;
cascading faults;
Byzantine-style perturbations.

These mechanisms are intended to model degradation or corruption of internal
feature representations and to evaluate whether dynamic topology control can
provide fault tolerance.

7. Computational latency

Inference latency is measured directly rather than assumed from a theoretical
constant.

The repository includes a latency benchmark for evaluating computational cost
under controlled feature-fault conditions.

Method overview

NeuroMesh modifies the bottleneck of a conventional multimodal U-Net.

                 Multimodal MRI
          ┌─────────────────────────┐
          │ T1 / T1ce / T2 / FLAIR │
          └────────────┬────────────┘
                       │
                       ▼
                U-Net Encoder
                       │
                       ▼
                 Bottleneck
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Feature representation     GCN–GRU Controller
                                      │
                                      ▼
                              Edge-activation mask
                                      │
                                      ▼
                          Dynamic feature topology
                                      │
                       ┌──────────────┘
                       ▼
                  U-Net Decoder
                       │
                       ▼
                 Segmentation

The topology controller combines:

feature projection;
recurrent state modeling through a GRU;
graph convolution;
mask prediction;
topology sparsity regularization;
topology diversity regularization.

The resulting edge-activation mask dynamically controls feature connectivity
at the bottleneck.

Main components
models/layers.py

Contains:

GraphConvLayer
NeuroMeshLayer

NeuroMeshLayer implements the MLP → GRU → GCN → mask-MLP topology-control
mechanism.

models/segmentation.py

Contains:

NeuroMeshUNet

This is the primary four-channel U-Net architecture with NeuroMesh topology
control at the bottleneck.

models/baselines.py

Contains comparison architectures:

PlainUNet
DropoutUNet
EnsembleUNet

These provide reference implementations for future systematic benchmarking.

utils/loss.py

Implements the composite NeuroMesh objective:

segmentation task loss;
rewiring regularization;
sparsity regularization;
diversity regularization.
utils/evaluation.py

Provides evaluation utilities including:

segmentation metrics;
modality-dropout evaluation;
latency measurement.
utils/failure_models.py

Implements controlled internal feature perturbations:

random;
correlated;
cascading;
Byzantine-style faults.
data/brats_loader.py

Provides loaders for:

adult BraTS NIfTI data;
pediatric BraTS-PEDs data;
pre-sliced adult HDF5 data;
synthetic MockBraTSDataset.

The loader also contains patient-level split utilities for avoiding slice-level
data leakage when working with pre-sliced datasets.

Adult and pediatric BraTS support

NeuroMesh is designed to handle both adult and pediatric BraTS data without
silently assuming that their data conventions are identical.

Adult BraTS

The loader supports the standard four-modal configuration:

T1
T1ce
T2
FLAIR

and handles common NIfTI naming/layout variations.

Adult segmentation labels are converted to the contiguous class representation
expected by the model.

Pediatric BraTS-PEDs

The pediatric loader supports:

T1n
T1c
T2w
T2f

and the corresponding pediatric segmentation convention.

The pediatric dataset contains a class structure that differs from the adult
BraTS formulation. NeuroMesh therefore does not silently collapse or remap
these classes without an explicit configuration choice.

Data policy

No patient or medical-imaging datasets are distributed with this repository.

The repository contains only code, figures, tests, and reproducibility
artifacts.

Real BraTS data must be obtained independently through the appropriate dataset
provider and used according to the applicable data-use agreement.

The .gitignore configuration excludes common medical-imaging formats and
local raw-data directories, including:

data/raw/
*.nii
*.nii.gz
*.h5
*.hdf5
*.dcm

This is intentional and prevents accidental redistribution of medical-imaging
data.

Reproducibility
Install dependencies
pip install -r requirements.txt
Run the complete test suite
pytest tests/ -v
Run a synthetic training experiment
python train.py --epochs 5 --n-train 32 --n-val 16
Generate the figures
python figures/fig_a_cortical_analogy.py
python figures/fig_b_dropout_panel.py
python figures/fig_c_latency_benchmark.py --budget-ms 100
Inspect a real adult BraTS HDF5 case
python data/brats_loader.py --inspect /path/to/volume_1_slice_75.h5
Figures

The figures/ directory contains scripts and generated figures documenting
the current implementation and validation workflow.

These include:

NeuroMesh topology/controller visualization;
modality-dropout experiments;
measured latency benchmarks;
real adult BraTS pipeline verification;
real pediatric BraTS-PEDs pipeline verification.

Figures generated from synthetic experiments are explicitly treated as
implementation demonstrations rather than evidence of real-world segmentation
performance.

Validation philosophy

NeuroMesh deliberately separates pipeline validation from scientific
performance validation.

Pipeline validation

The current release establishes that:

code
 ↓
data loading
 ↓
preprocessing
 ↓
model
 ↓
dynamic topology control
 ↓
fault injection
 ↓
evaluation

can be executed reproducibly.

This includes both synthetic experiments and real-data verification.

Scientific performance validation

A full real-data benchmark requires:

a sufficiently large real BraTS training cohort;
patient-level train/validation/test separation;
real-data model training;
hyperparameter selection without test-set leakage;
baseline training under comparable conditions;
independent test-set evaluation;
repeated runs/seeds;
statistical analysis;
systematic modality-dropout experiments;
systematic fault-injection experiments.

These experiments constitute the next stage of NeuroMesh development.

Future work

Future development will focus on systematic real-data training and evaluation.

Planned work includes:

Training NeuroMesh on properly partitioned real BraTS cohorts.
Establishing patient-level train/validation/test protocols.
Comparing NeuroMesh against U-Net, stochastic/ensemble baselines, and
established medical-image segmentation frameworks.
Performing ablation studies of the GCN, GRU, rewiring, sparsity, and
diversity components.
Quantifying segmentation robustness under increasing modality and
feature-level fault severity.
Measuring topology adaptation and active-edge behavior during faults.
Evaluating computational overhead and deployment efficiency.
Extending evaluation across adult and pediatric populations.
Investigating external-cohort generalization.
Exploring clinically relevant validation only after sufficient
methodological and real-data evidence has been established.

The current work is therefore intended as the foundation for a subsequent
systematic real-data study rather than as a clinical validation study.

Relationship to existing frameworks

The implementation of losses and evaluation metrics was cross-checked against
established medical-imaging software where applicable.

MONAI was used as a reference for
metric/loss verification.

nnU-Net is identified as an important
reference framework for future real-data benchmarking.

Neither framework is vendored into this repository.

Repository structure
NeuroMesh/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── data/
│   ├── __init__.py
│   └── brats_loader.py
│
├── figures/
│   ├── fig_a_cortical_analogy.py
│   ├── fig_b_dropout_panel.py
│   ├── fig_c_latency_benchmark.py
│   ├── fig_a_cortical_analogy.png
│   ├── fig_b_dropout_panel.png
│   ├── fig_c_latency_benchmark.png
│   ├── fig_peds_case_sanity_check.png
│   └── fig_real_case_sanity_check.png
│
├── models/
│   ├── __init__.py
│   ├── baselines.py
│   ├── layers.py
│   └── segmentation.py
│
├── paper/
│   └── media/
│       └── figures/
│
├── tests/
│   ├── __init__.py
│   └── test_neuromesh.py
│
├── utils/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── failure_models.py
│   └── loss.py
│
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── train.py
Citation

A formal publication citation will be added after peer-reviewed publication.

If you use NeuroMesh before publication, please cite the GitHub repository and
identify the software version/commit used in your work.

BraTS dataset

NeuroMesh uses and supports BraTS datasets. Users must cite the appropriate
BraTS dataset publications corresponding to the specific dataset version used.

License

NeuroMesh is released under the MIT License.

See LICENSE for the complete license text.

The license applies to the software contained in this repository and does not
grant redistribution rights for third-party medical-imaging datasets.

```
