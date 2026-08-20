# Installation

```bash
git clone <this-repo>
cd NeuroMesh
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e ".[all]"
```

Extras: `.[data]` (nibabel, h5py — needed for real dataset loading), `.[viz]` (matplotlib — needed for figure scripts), `.[dev]` (pytest, ruff).

Verify:

```bash
python -c "import neuromesh; print(neuromesh.__version__)"
neuromesh --help
pytest tests/
```

## GPU

Not required. This pilot's own experiments ran entirely on CPU (1 core, no GPU — see `reproducibility.md`). Install a CUDA-enabled PyTorch build separately per [pytorch.org](https://pytorch.org) if you have a GPU available; `neuromesh train --device cuda` will use it automatically when present (`torch.cuda.is_available()`).

## Common issues

See `troubleshooting.md`.
