# Contributing

This is a research repository documenting a specific pilot study. Contributions are welcome for:
- Bug fixes (please add a regression test, following the pattern in `tests/regression/`)
- Documentation improvements
- Additional baselines or analysis tooling, clearly separated from the frozen v1 protocol's results

**Do not** modify files under `results/frozen/` or `PROTOCOL_FREEZE_v1.md` to "improve" reported numbers. If you re-run an experiment, save it as a new, clearly-versioned result (e.g. a `v2` protocol) rather than overwriting v1.

## Development setup

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
```

## Before submitting a PR

- `pytest tests/` passes.
- New functionality has tests (synthetic fixtures only -- CI does not have access to real patient data).
- Any new result file has a clear provenance entry in `results/manifests/RESULTS_MANIFEST.md`.
