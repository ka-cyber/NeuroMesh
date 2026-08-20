#!/usr/bin/env python3
"""Thin wrapper: `python scripts/train_real.py <args>` == `neuromesh train <args>`.
Prefer the installed `neuromesh` CLI directly; this exists for people running
from a repo checkout without having run `pip install -e .` yet."""
import sys
from neuromesh.training.train_real import main

if __name__ == "__main__":
    main()
