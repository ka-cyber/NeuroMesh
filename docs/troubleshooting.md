# Troubleshooting

**`ModuleNotFoundError: No module named 'neuromesh'`**
Run `pip install -e .` from the repository root (the directory containing `pyproject.toml`).

**`ImportError: nibabel is required` / `h5py` missing**
Install the data extra: `pip install -e ".[data]"`.

**GRUCell / hidden-state batch-size errors during training or evaluation**
This was a real bug (see README, `tests/regression/`). If you see it, you're likely running against a modified `evaluate_modality_dropout` or a custom training loop that dropped the batch-size-mismatch guard — check for `if h_state is not None and h_state.size(0) != x.size(0): h_state = None` in your code path.

**Training seems to hang / is extremely slow**
Expected at native 240x240 resolution without a GPU — this pilot used `--image-size 96` specifically for this reason (see `reproducibility.md`). Use `--max-seconds` to checkpoint/resume in a time-limited environment rather than waiting for one call to finish.

**`kaggle datasets download` fails / 403**
Confirm `~/.kaggle/kaggle.json` (or `KAGGLE_API_TOKEN`) is set with a valid, non-expired token from your Kaggle account settings.

**Frozen test set results look wrong / I want to re-run test evaluation**
Don't, without first reading `PROTOCOL_FREEZE_v1.md` and updating it. Re-evaluating a new/changed model against the same frozen test patients without recording that decision defeats the purpose of freezing it.
