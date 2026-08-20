#!/usr/bin/env python3
"""
Regenerates the patient-level FLAIR-missing WT bar chart from stored results
(results/validation/*.json) -- the one figure this pilot's real results
support well as a bar chart. For the mask/gate topology figures, use
`neuromesh analyze-controller` directly (it renders per-condition heatmaps
from a live checkpoint, not from a pre-aggregated JSON, since the mask
arrays are stored separately as .npz).
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=".")
    p.add_argument("--out", default="figures/patient_level_flair_wt.png")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib required -- pip install -e '.[viz]'")

    model_paths = {
        "NeuroMesh (e4)": "results/validation/VAL_neuromesh_e4_regions.json",
        "Plain UNet": "results/validation/VAL_plain_real30_regions.json",
        "DropoutUNet": "results/validation/VAL_dropout_real30_regions.json",
        "StaticGatedUNet": "results/validation/VAL_staticgate_real30_regions.json",
    }
    data = {}
    for name, rel in model_paths.items():
        path = os.path.join(args.results_dir, rel)
        if not os.path.exists(path):
            print(f"[generate_figures] MISSING: {path} -- skipping {name}")
            continue
        data[name] = json.load(open(path))

    patients = sorted(next(iter(data.values()))["results"]["FLAIR_missing"]["per_patient"].keys(), key=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(data)
    for i, (name, d) in enumerate(data.items()):
        vals = [d["results"]["FLAIR_missing"]["per_patient"][p]["WT"]["dice"] for p in patients]
        xs = [j + i * width for j in range(len(patients))]
        ax.bar(xs, vals, width=width, label=name)
    ax.set_xticks([j + width * (len(data) - 1) / 2 for j in range(len(patients))])
    ax.set_xticklabels([f"patient {p}" for p in patients])
    ax.set_ylabel("WT Dice under FLAIR-missing")
    ax.set_title("Patient-level WT collapse under FLAIR loss (val, n=4, pilot)")
    ax.legend()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[generate_figures] saved -> {args.out}")


if __name__ == "__main__":
    main()
