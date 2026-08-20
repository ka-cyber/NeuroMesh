"""
NeuroMesh: a topology-inspired segmentation controller for multimodal MRI,
studied here specifically for behavior under missing-modality conditions.

This package is the real-data research implementation. See
docs/architecture.md for the important distinction this project draws
between the INTENDED mechanism (input-conditional topology rewiring) and the
OBSERVED mechanism in our pilot experiments (a largely static learned
bottleneck gate) -- that distinction is the central empirical finding of
this repository, not an implementation detail to gloss over.
"""

__version__ = "0.1.0"
