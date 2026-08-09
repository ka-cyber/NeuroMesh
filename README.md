NeuroMesh

NeuroMesh is a research-oriented deep learning project for medical image segmentation, with support for BraTS-style brain tumor imaging datasets.

Research software: NeuroMesh is intended for research and experimentation. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

Repository Structure

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

Overview

The repository is organized into the following components:

data/ — Dataset loading and preprocessing utilities, including BraTS support.

models/ — NeuroMesh model components, layers, segmentation models, and baselines.

utils/ — Loss functions, evaluation utilities, and failure-modeling components.

figures/ — Figure-generation scripts and generated figures used for analysis and paper preparation.

tests/ — Automated tests for the NeuroMesh implementation.

train.py — Main training entry point.

.github/workflows/ — Continuous integration workflows.

Dockerfile — Containerized environment for reproducible execution.

paper/ — Supporting publication media and figures.

Requirements

Install the dependencies listed in requirements.txt:

git clone https://github.com/<your-username>/NeuroMesh.git
cd NeuroMesh

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

On Windows PowerShell:

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

Dataset

NeuroMesh uses and supports BraTS datasets.

Because different BraTS releases may have different imaging modalities, labels, preprocessing conventions, and access conditions, make sure that the dataset version used in an experiment is compatible with the data loader and evaluation pipeline.

Dataset Citation

Users must cite the appropriate BraTS dataset publication(s) corresponding to the specific dataset version used in their work.

Important: BraTS is a third-party medical-imaging dataset. Its terms of use, access requirements, attribution requirements, and redistribution restrictions are separate from the NeuroMesh software license.

Data Preparation

Place the BraTS dataset in the location expected by data/brats_loader.py, or update the dataset path/configuration used by the training pipeline.

Before training, verify that:

The dataset was obtained from the appropriate official source.

You comply with the dataset's terms of use.

Image modalities and labels follow the expected format.

Training, validation, and test splits do not contain unintended data leakage.

Training

The main training entry point is:

python train.py

If command-line arguments are supported, inspect them with:

python train.py --help

For reproducible experiments, record:

dataset name and version

dataset split

model configuration

training configuration

random seed

Python version

dependency versions

hardware configuration

NeuroMesh Git commit

Evaluation

Evaluation utilities are located in:

utils/evaluation.py

Model implementations are located in:

models/

The repository also contains analysis and sanity-check figures in:

figures/

These include:

cortical analogy analysis

dropout analysis

latency benchmarking

pediatric-case sanity checks

real-case sanity checks

Testing

Run the test suite with:

pytest

or:

python -m pytest

GitHub Actions testing is configured in:

.github/workflows/tests.yml

Docker

A Dockerfile is included for reproducible containerized execution.

Build the Docker image:

docker build -t neuromesh .

Run the container:

docker run --rm -it neuromesh

If GPU acceleration is required, configure the appropriate Docker GPU runtime for your environment.

Figures and Paper Materials

The figures/ directory contains scripts and generated figures used for analysis and research communication:

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

Reproducibility

For research use, we recommend recording the following information for every experiment:

NeuroMesh Git commit or software version

Python version

dependency versions

BraTS dataset name and version

dataset split

model configuration

training configuration

random seed

hardware configuration

evaluation configuration

This information helps reproduce reported results and compare experiments fairly.

Citation

A formal publication citation will be added after peer-reviewed publication.

If you use NeuroMesh before publication, please cite the GitHub repository and identify the software version or commit used in your work.

Example:

NeuroMesh, GitHub repository, version/commit <VERSION_OR_COMMIT>,
accessed <DATE>.

When a peer-reviewed publication becomes available, please use the formal citation provided by the project.

BraTS Dataset Citation

NeuroMesh uses and supports BraTS datasets. Users must cite the appropriate BraTS dataset publication(s) corresponding to the specific dataset version used in their work.

License

NeuroMesh is released under the MIT License.

See LICENSE for the complete license text.

The license applies to the software contained in this repository and does not grant redistribution rights for third-party medical-imaging datasets, including BraTS data.

Users are responsible for complying with all applicable dataset licenses, access agreements, and citation requirements.

Disclaimer

NeuroMesh is research software provided for experimentation and scientific research.

It has not been presented in this repository as a clinically validated medical device. The software must not be relied upon for clinical diagnosis, treatment decisions, or other medical decisions.

Contributing

Contributions, bug reports, reproducibility reports, and research feedback are welcome.

When opening an issue or pull request, please provide enough information to reproduce the problem, including relevant:

Python and dependency versions

operating system

dataset/version

configuration

error messages or logs

Git commit

Contact

For questions, issues, or research collaboration, please use the repository's GitHub Issues page or the contact information provided by the project maintainers.

NeuroMesh — research software for medical image segmentation.
