NeuroMeshNeuroMesh is a self-reconfiguring graph-neural topology controller for fault-tolerant multimodal medical image segmentation.The framework integrates a GCN–GRU topology controller into the bottleneck of a four-channel U-Net to dynamically predict an edge-activation mask over feature representations. The controller is trained using a composite objective combining segmentation performance with topology-rewiring, sparsity, and diversity regularization.NeuroMesh is designed to investigate whether dynamic computational topology reconfiguration can provide increased robustness to modality degradation and internal feature faults in multimodal medical image segmentation.📌 Project StatusEngineering & Implementation — CompleteThe current release contains a complete, executable implementation of the NeuroMesh framework:Tests: 26 / 26 automated unit and integration tests passing.Pipeline: Complete end-to-end training and evaluation executed on synthetic data.Real Data Ingestion: Adult BraTS2021 and pediatric BraTS-PEDs datasets successfully loaded, processed, and evaluated using real segmentation labels for single-case verification.Fault Simulation: Synthetic modality-dropout and internal feature-fault experiments implemented.Benchmarking & Baselines: Latency benchmarks and baseline architectures included.Reproducibility: Configured via requirements.txt, Docker, and GitHub Actions CI. Included figure-generation scripts.Scientific Validation — Deliberately ScopedThe current release is a methodological and implementation-validation release, not a population-level clinical study.The checkpoint used for single-case verification experiments was trained on synthetic data. Therefore, single-case real-data verification must not be interpreted as an estimate of real-world segmentation performance or generalization.⚠️ Non-Claims (Current Release Scope):
 ✖ State-of-the-art BraTS segmentation performance
 ✖ Population-level Dice or HD95 benchmarks
 ✖ Superiority over nnU-Net or established baselines
 ✖ Clinical diagnostic utility or clinical readiness
🏗️ Method OverviewNeuroMesh modifies the bottleneck of a conventional multimodal U-Net to dynamically manage feature connectivity during inference and feature corruption.                  Multimodal MRI Inputs
              ┌───────────────────────────┐
              │  T1 / T1ce / T2 / FLAIR   │
              └─────────────┬─────────────┘
                            │
                            ▼
                      U-Net Encoder
                            │
                            ▼
                        Bottleneck
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    Feature Representation      GCN–GRU Controller
              │                           │
              │                           ▼
              │                  Edge-Activation Mask
              │                           │
              ▼                           ▼
       └──────────────► Dynamic Feature Topology
                                  │
                                  ▼
                            U-Net Decoder
                                  │
                                  ▼
                        Segmentation Output
Key Controller MechanicsFeature Projection: Maps compressed bottleneck maps to graph node embeddings.Recurrent State (GRU): Retains temporal/sequential topology dynamics across passes.Graph Convolution (GCN): Computes relational interactions across spatial feature regions.Mask Prediction & Regularization: Predicts an edge-activation mask governed by rewiring, sparsity, and diversity loss terms.🔍 Validation Suite1. Software & Unit-Level ValidationAutomated tests covering model architectures, loss formulations, evaluation tools, and data-loading routines:Bash26 / 26 automated tests passing
2. Synthetic PipelineA dependency-light synthetic dataset (MockBraTSDataset) exercises the complete data flow:$$\text{Data Gen} \rightarrow \text{Multimodal Input} \rightarrow \text{Encoder} \rightarrow \text{NeuroMesh Bottleneck} \rightarrow \text{Decoder} \rightarrow \text{Loss/Eval}$$3. Real Adult & Pediatric VerificationAdult BraTS2021: Validated against 4 standard modalities (T1, T1ce, T2, FLAIR) and real labels.Pediatric BraTS-PEDs: Validated against pediatric naming conventions (T1n, T1c, T2w, T2f) and specific class mapping structures without silent remapping.4. Fault & Latency ModelingModality Dropout: Simulates missing input channels dynamically at inference time.Feature Faults: Simulates internal network feature disruptions (Random, Correlated, Cascading, Byzantine-style perturbations).Latency Benchmarking: Directly measures inference time under various fault conditions.📂 Repository StructurePlaintextNeuroMesh/
│
├── .github/
│   └── workflows/
│       └── tests.yml           # CI/CD pipeline configuration
│
├── data/
│   ├── __init__.py
│   └── brats_loader.py         # Loaders for adult/pediatric BraTS & synthetic mock data
│
├── figures/                    # Scripts and outputs for paper/doc visualizations
│   ├── fig_a_cortical_analogy.py
│   ├── fig_b_dropout_panel.py
│   ├── fig_c_latency_benchmark.py
│   └── *.png
│
├── models/
│   ├── __init__.py
│   ├── baselines.py            # PlainUNet, DropoutUNet, EnsembleUNet
│   ├── layers.py               # GraphConvLayer & NeuroMeshLayer (MLP->GRU->GCN->MLP)
│   └── segmentation.py         # NeuroMeshUNet architecture
│
├── tests/
│   └── test_neuromesh.py       # Pytest suite
│
├── utils/
│   ├── evaluation.py           # Metrics, latency, & dropout scripts
│   ├── failure_models.py       # Random, Correlated, Cascading, Byzantine fault injection
│   └── loss.py                 # Composite segmentation + topology loss
│
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── train.py                    # Main training and execution entrypoint
⚡ Quickstart & Reproducibility1. Environment SetupBash# Clone the repository
git clone https://github.com/your-org/NeuroMesh.git
cd NeuroMesh

# Install dependencies
pip install -r requirements.txt
2. Run Test SuiteBashpytest tests/ -v
3. Synthetic Pipeline ExecutionBashpython train.py --epochs 5 --n-train 32 --n-val 16
4. Generate Figures & BenchmarksBashpython figures/fig_a_cortical_analogy.py
python figures/fig_b_dropout_panel.py
python figures/fig_c_latency_benchmark.py --budget-ms 100
5. Inspect Real Dataset SlicesBashpython data/brats_loader.py --inspect /path/to/volume_1_slice_75.h5
🔐 Data Policy & PrivacyNo Medical Data Included: This repository contains only code, tests, and configuration files.Data Access: Real BraTS datasets must be obtained independently through official BraTS Challenges under their respective Data Use Agreements.Git Safety: The .gitignore is strictly configured to exclude common medical imaging formats (*.nii, *.nii.gz, *.h5, *.hdf5, *.dcm) and raw data directories (data/raw/).🗺️ Roadmap & Future WorkThe current release serves as the foundation for an upcoming systematic study:[ ] Train NeuroMesh on full, properly partitioned real BraTS cohorts with patient-level splitting.[ ] Systematic benchmarking against nnU-Net, standard U-Net, and stochastic/ensemble baselines.[ ] Comprehensive ablation studies of GCN, GRU, rewiring, sparsity, and diversity components.[ ] Quantitative evaluation of topological adaptation under increasing feature fault severity.[ ] Exploration of multi-site external cohort generalization.🤝 References & AcknowledgmentsCross-Verification: Evaluation metrics and loss function behaviors were verified against implementations in MONAI.Benchmarking Reference: Architecture and preprocessing protocols reference nnU-Net design principles. (Neither framework is vendored within this codebase).📜 License & CitationLicenseNeuroMesh is released under the MIT License. The license applies strictly to the software in this repository and does not cover external datasets.CitationA formal paper citation will be added upon peer-reviewed publication. If you use NeuroMesh in your research prior to publication, please cite this repository:Code snippet@software{neuromesh2026,
  author = {NeuroMesh Contributors},
  title = {NeuroMesh: Self-Reconfiguring Graph-Neural Topology Controller for Fault-Tolerant Medical Segmentation},
  url = {https://github.com/ka-cyber/NeuroMesh},
  year = {2026}
}
Note: When publishing work using BraTS data, ensure you cite the appropriate BraTS dataset papers corresponding to the specific challenge year used.
