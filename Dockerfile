# NeuroMesh -- reproducible development/training environment.
#
# This gives you a consistent CPU environment for running tests and the
# synthetic-data pipeline anywhere. It does NOT include CUDA/GPU support --
# for real training you almost certainly want a GPU. Two ways to get that:
#   1. Start FROM a CUDA base image instead (e.g. nvidia/cuda:12.4.0-devel-ubuntu22.04)
#      and install torch with CUDA support per https://pytorch.org/get-started/locally/
#   2. Use this CPU image for development/testing only, and train on a
#      separate GPU machine / cloud instance / Colab using the same
#      requirements.txt.

FROM python:3.12-slim

WORKDIR /workspace

# System deps: nibabel/h5py need HDF5; matplotlib needs some font libs at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Sanity check at build time: fail the image build if the package doesn't
# even import cleanly, instead of discovering it at `docker run` time.
RUN python -c "from models.segmentation import NeuroMeshUNet; from data.brats_loader import MockBraTSDataset; print('NeuroMesh package OK')"

CMD ["python", "-m", "pytest", "tests/", "-v"]
