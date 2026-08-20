# Results manifest

Every file below is preserved exactly as originally produced — no numbers were regenerated, rounded, or edited when this repository was assembled. Checksums: `RESULTS_CHECKSUMS.sha256` in this directory.

## `results/frozen/` — touched exactly once per model, per `PROTOCOL_FREEZE_v1.md`

| file | checkpoint | test patients | protocol |
|---|---|---|---|
| `FROZEN_TEST_neuromesh_real30.json` | `neuromesh_real30.pt` (4 epochs) | 48, 53, 115, 302 | `PROTOCOL_FREEZE_v1.md` |
| `FROZEN_TEST_plain_real30.json` | `plain_real30.pt` (4 epochs) | 48, 53, 115, 302 | same protocol, same split |

DropoutUNet and StaticGatedUNet were **not** evaluated on the frozen test set as of this repository snapshot — per explicit instruction to keep the test set frozen until the full model set/protocol is finalized. Their results in `results/validation/` are val-only.

## `results/validation/` — safe to have been regenerated during development

| file | model | epochs | notes |
|---|---|---|---|
| `VAL_neuromesh_e4_regions.json` | NeuroMesh | 4 | WT/TC/ET/HD95/sensitivity/precision, matched split to test |
| `VAL_neuromesh_e8_regions.json` | NeuroMesh | 8 | continued training, same 22 train patients, optimizer state reset at the 4-epoch boundary (see PROTOCOL_FREEZE_v1.md) |
| `VAL_plain_real30_regions.json` | Plain UNet | 4 | |
| `VAL_dropout_real30_regions.json` | DropoutUNet | 4 | |
| `VAL_staticgate_real30_regions.json` | StaticGatedUNet | 4 | |
| `real_*_results.json` | (various) | — | earlier-format results using raw per-class Dice (not WT/TC/ET composite regions) — superseded by the `VAL_*_regions.json` files above for scientific reporting, kept for provenance |

## `results/mechanism/`

| directory | split | notes |
|---|---|---|
| `epoch4_TOUCHED_TEST_SET/` | **test** (patients 48, 53, 115, 302) | Run before the `--split` flag existed; defaulted to test. This was diagnostic controller inspection *after* the official epoch-4 test evaluation of these same models had already completed under the frozen protocol — not used for any model-selection decision. Disclosed here rather than hidden. |
| `epoch8_val_only/` | val (patients 112, 126, 259, 333) | Explicitly restricted to val once the leakage risk above was identified. |

## Regenerating tables from these files

```bash
neuromesh compare --results-dir . --out-dir results/manifests
```

Produces `four_model_validation.{csv,md}` and `patient_level_FLAIR_missing_WT.{csv,md}` — the exact tables in the README, generated fresh from the JSON above every time. If a reported number in the README ever looks wrong, regenerate these and diff — the JSON files are the source of truth, not the README prose.
