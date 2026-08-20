#!/usr/bin/env python3
"""Thin wrapper: `python scripts/analyze_controller.py <args>` == `neuromesh analyze-controller <args>`."""
from neuromesh.analysis.mechanism_analysis import main

if __name__ == "__main__":
    main()
