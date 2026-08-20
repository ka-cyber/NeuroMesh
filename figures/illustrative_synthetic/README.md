# Illustrative / pre-real-data figures

**None of the figures in this directory represent the 30-patient real-data pilot study.** They predate it. Kept for historical/illustrative reference only -- do not cite these as scientific results.

| file | what it actually is |
|---|---|
| `fig_a_cortical_analogy.png` | Conceptual/illustrative diagram of the architecture's motivating analogy. Not a result. |
| `fig_b_dropout_panel.png` | Generated against `MockBraTSDataset` (synthetic data). Demonstrates the evaluation pipeline runs end to end -- the Dice numbers in it are meaningless clinically. |
| `fig_c_latency_benchmark.png` | Same -- synthetic-data latency benchmark, pipeline-check only. |
| `fig_real_case_sanity_check.png` | The one figure here that touched *real* MRI data -- a single real BraTS patient (source: a separate, earlier raw-NIfTI download, NOT part of the 30-patient Kaggle-subset study), used only to verify the data-loading pipeline parses real files correctly before the actual 30-patient study began. The near-zero Dice shown is expected and intentional (the model shown was trained only on synthetic data) -- it is a wiring check, not a performance result. |

For the real study's actual figures/tables, see `results/manifests/` and `neuromesh compare` / `neuromesh analyze-controller` output.
