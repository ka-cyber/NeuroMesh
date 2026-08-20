# Security Policy

This is a research repository; it does not process live patient data or accept untrusted network input by design. If you find a security issue (e.g. unsafe deserialization, path traversal in data loading, a leaked credential in git history), please open a private report via GitHub's security advisory feature rather than a public issue.

Note: `torch.load(..., weights_only=False)` is used in several places to load checkpoints that include non-tensor metadata (args dicts, step counters). Only load checkpoints from sources you trust -- this is standard PyTorch checkpoint-loading risk, not specific to this codebase.
