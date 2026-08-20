#!/usr/bin/env python3
"""Thin wrapper: `python scripts/evaluate.py <args>` == `neuromesh evaluate <args>`."""
from neuromesh.evaluation.evaluate_frozen_test import main

if __name__ == "__main__":
    main()
