# NeuroMesh -- reproducible development/training environment.
#
# CPU-only. This pilot's own reported experiments ran on CPU (see
# docs/reproducibility.md) -- for real training at scale you almost
# certainly want a GPU. Two options:
#   1. Start FROM a CUDA base image instead (e.g. nvidia/cuda:12.4.0-devel-ubuntu22.04)
#      and install torch with CUDA support per https://pytorch.org/get-started/locally/
#   2. Use this CPU image for development/testing only, and train on a
#      separate GPU machine / cloud instance using the same pyproject.toml.
#
# NOTE: this Dockerfile has not been verified to build in the environment
# this repository was assembled in (no Docker daemon was available there).
# Review before relying on it in production.

FROM python:3.12-slim

WORKDIR /workspace

# System deps: nibabel/h5py need HDF5; matplotlib needs font libs at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY neuromesh/ neuromesh/
COPY README.md .

RUN pip install --no-cache-dir -e ".[all]"

COPY . .

# Fail the build if the package doesn't even import cleanly, rather than
# discovering it at `docker run` time.
RUN python -c "from neuromesh.models import NeuroMeshUNet; from neuromesh.data import MockBraTSDataset; print('NeuroMesh package OK')" \
    && neuromesh --help

CMD ["pytest", "tests/", "-v"]
