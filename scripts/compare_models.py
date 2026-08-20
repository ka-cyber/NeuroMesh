#!/usr/bin/env python3
"""Thin wrapper: `python scripts/generate_tables.py <args>` == `neuromesh compare <args>`.
Regenerates every table in README.md from results/ JSON -- no new computation."""
from neuromesh.analysis.compare import main

if __name__ == "__main__":
    main()
