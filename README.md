NeuroMesh

NeuroMesh is a research-oriented deep learning project for medical imagesegmentation, with support for BraTS-style brain tumor imaging datasets.The repository contains model architectures, baseline implementations,data loading utilities, training and evaluation code, tests,reproducible figures, and supporting paper media.

Research software: NeuroMesh is intended for research andexperimentation. It is not a medical device and must not be used forclinical diagnosis or treatment decisions.

Repository Structure

NeuroMesh/
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

Overview

The repository is organized around four main components:

data/ --- Dataset loading and preprocessing utilities,including BraTS support.

models/ --- NeuroMesh model components, segmentation models,layers, and baseline architectures.

utils/ --- Loss functions, evaluation utilities, andfailure-modeling components.

figures/ --- Scripts and generated figures used for analysis,benchmarking, and paper preparation.

tests/ --- Automated tests for the NeuroMesh implementation.

train.py --- Main training entry point.

.github/workflows/ --- Continuous integration workflows.

Dockerfile --- Containerized environment for reproducibleexecution.

paper/ --- Supporting media and publication figures.

Requirements

NeuroMesh requires Python and the dependencies listed inrequirements.txt.

To install the dependencies:

git clone https://github.com/<your-username>/NeuroMesh.git
cd NeuroMesh

python -m venv .venv
source .venv/bin/activate

On Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

Then install the project dependencies:

pip install --upgrade pip
pip install -r requirements.txt

Dataset

NeuroMesh uses and supports BraTS datasets.

Because BraTS dataset releases may differ in imaging modalities, labels,preprocessing conventions, and licensing/access conditions, make surethat the dataset version used in an experiment matches the assumptionsof the corresponding data loader and evaluation pipeline.

Dataset Citation

Users must cite the appropriate BraTS dataset publication(s)corresponding to the specific BraTS release used in their work.

Important: BraTS data are third-party datasets. Their terms ofaccess, use, attribution, and redistribution are separate from thelicense of the NeuroMesh source code.

Data Preparation

Place the dataset in the location expected by data/brats_loader.py, orupdate the dataset configuration/path used by your training setup.

Before training, verify:

The dataset is downloaded from the appropriate official source.

You have accepted and comply with the dataset's terms of use.

Image modalities and label files follow the expected naming/formatconventions.

The dataset split does not introduce unintendedtrain/validation/test leakage.

Training

The main training entry point is:

python train.py

If train.py exposes command-line arguments, inspect the availableoptions with:

python train.py --help

For reproducible experiments, record the dataset version, configuration,random seed, software environment, and Git commit used for the run.

Evaluation

Evaluation utilities are located in:

utils/evaluation.py

Model-specific code and segmentation implementations are located in:

models/

The repository also contains analysis and sanity-check figures under:

figures/

These include:

cortical analogy analysis

dropout analysis

latency benchmarking

pediatric-case sanity checks

real-case sanity checks

Testing

Run the test suite with:

python -m pytest

or:

pytest

The repository also includes a GitHub Actions workflow at:

.github/workflows/tests.yml

to support automated testing in CI.

Docker

A Dockerfile is provided for a reproducible containerized environment.

Build the image:

docker build -t neuromesh .

Run the container:

docker run --rm -it neuromesh

Depending on your training configuration and hardware requirements, youmay need additional Docker or GPU runtime options.

Reproducibility

For research use, we recommend recording at least:

NeuroMesh Git commit or release version

Python version

dependency versions

BraTS dataset name and version

dataset split

model configuration

training configuration

random seed

hardware configuration

evaluation configuration

This information makes it easier to reproduce reported results andcompare experiments fairly.

Figures and Paper Materials

The figures/ directory contains both figure-generation scripts andgenerated figures used for analysis and research communication.

figures/
├── fig_a_cortical_analogy.py
├── fig_b_dropout_panel.py
├── fig_c_latency_benchmark.py
├── fig_a_cortical_analogy.png
├── fig_b_dropout_panel.png
├── fig_c_latency_benchmark.png
├── fig_peds_case_sanity_check.png
└── fig_real_case_sanity_check.png

Additional publication media are stored under:

paper/media/figures/

Citation

A formal publication citation will be added after peer-reviewedpublication.

If you use NeuroMesh before publication, please cite the GitHubrepository and identify the software version or commit used in yourwork.

For example:

NeuroMesh, GitHub repository, version/commit <VERSION_OR_COMMIT>,
accessed <DATE>.

When a peer-reviewed publication becomes available, please use theformal citation provided by the project.

BraTS Citation

NeuroMesh uses and supports BraTS datasets. Users must cite theappropriate BraTS dataset publication(s) corresponding to the specificdataset version used in their work.

License

NeuroMesh is released under the MIT License.

See LICENSE for the complete license text.

The MIT License applies to the software contained in this repository. Itdoes not grant redistribution rights for third-party medical-imagingdatasets, including BraTS data. Users are responsible for complying withthe applicable dataset licenses, access agreements, and citationrequirements.

Disclaimer

NeuroMesh is research software provided for experimentation andscientific research. It has not been presented in this repository as aclinically validated medical device.

The authors and contributors make no guarantees regarding clinicalperformance, diagnostic accuracy, treatment recommendations, orsuitability for any particular medical purpose.

Contributing

Contributions, bug reports, reproducibility reports, and researchfeedback are welcome.

When opening an issue or pull request, please provide enough informationto reproduce the problem, including relevant:

Python and dependency versions

operating system

dataset/version

configuration

error messages or logs

Git commit

Contact

For questions, issues, or research collaboration, please use therepository's GitHub Issues page or the contact information provided bythe project maintainers.

NeuroMesh --- research software for medical image segmentation.
