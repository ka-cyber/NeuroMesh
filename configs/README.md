# Configs

`experiments/` -- one YAML per trained model, documenting the exact hyperparameters used (mirrors `PROTOCOL_FREEZE_v1.md`, which is authoritative if these ever disagree). Not currently auto-consumed by `neuromesh train` (which is CLI-flag driven) -- these exist for documentation/provenance, and as a template if you want to add YAML-config support later.

`data/` -- dataset source/format documentation, cross-referenced with `docs/dataset.md`.
