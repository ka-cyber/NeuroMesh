"""
neuromesh/analysis/compare.py

Regenerates the manuscript/README-facing comparison tables STRICTLY from
already-saved result JSON files (results/validation/, results/frozen/,
results/mechanism/). This module performs no training, no evaluation, and no
new computation beyond arithmetic already implicit in the stored numbers
(e.g. formatting mean +/- std). If a result file is missing, the
corresponding table cell is reported as missing, not invented.

This exists because raw numbers should never be hand-copied into a README or
a manuscript table -- every reported number needs a script that can
regenerate it from the JSON on disk, so a reviewer (or a future you) can
check provenance.
"""
import argparse
import csv
import json
import os
from pathlib import Path


CONDITIONS = ["clean", "FLAIR_missing", "T1_missing", "T1ce_missing", "T2_missing"]
REGIONS = ["WT", "TC", "ET"]


def load_result(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def four_model_table(model_result_paths, conditions=CONDITIONS):
    """
    model_result_paths: dict of {model_name: path_to_regions_json}
    Returns a list of rows: [condition, model, WT_mean, WT_std, TC_mean, TC_std, ET_mean, ET_std]
    Missing files/models are reported as 'MISSING', never silently skipped or
    filled in with a plausible-looking placeholder.
    """
    loaded = {name: load_result(path) for name, path in model_result_paths.items()}
    rows = []
    for cond in conditions:
        for name, data in loaded.items():
            if data is None:
                rows.append([cond, name] + ["MISSING"] * 6)
                continue
            s = data["results"][cond]["summary"]
            row = [cond, name]
            for region in REGIONS:
                m = s[region]["dice"]["mean"]
                sd = s[region]["dice"]["std"]
                row += [f"{m:.4f}" if m is not None else "undefined",
                        f"{sd:.4f}" if sd is not None else "undefined"]
            rows.append(row)
    return rows


def patient_level_table(model_result_paths, condition, region="WT"):
    """One condition/region, every patient, every model -- the granularity
    your protocol explicitly requires (never aggregate-only)."""
    loaded = {name: load_result(path) for name, path in model_result_paths.items()}
    all_patients = set()
    for data in loaded.values():
        if data is not None:
            all_patients |= set(data["results"][condition]["per_patient"].keys())
    patients = sorted(all_patients, key=int)

    rows = []
    for pid in patients:
        row = [pid]
        for name, data in loaded.items():
            if data is None or pid not in data["results"][condition]["per_patient"]:
                row.append("MISSING")
            else:
                row.append(f"{data['results'][condition]['per_patient'][pid][region]['dice']:.4f}")
        rows.append(row)
    return list(loaded.keys()), rows


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_markdown(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


DEFAULT_MODEL_PATHS = {
    "NeuroMesh (e4)": "results/validation/VAL_neuromesh_e4_regions.json",
    "Plain UNet": "results/validation/VAL_plain_real30_regions.json",
    "DropoutUNet": "results/validation/VAL_dropout_real30_regions.json",
    "StaticGatedUNet": "results/validation/VAL_staticgate_real30_regions.json",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=".", help="Repo root containing results/validation/")
    p.add_argument("--out-dir", default="results/manifests")
    p.add_argument("--condition", default="FLAIR_missing",
                    help="Condition for the patient-level table (default: the diagnostic FLAIR_missing case)")
    args = p.parse_args()

    root = Path(args.results_dir)
    model_paths = {name: str(root / rel) for name, rel in DEFAULT_MODEL_PATHS.items()}

    header = ["condition", "model", "WT_mean", "WT_std", "TC_mean", "TC_std", "ET_mean", "ET_std"]
    rows = four_model_table(model_paths)
    write_csv(os.path.join(args.out_dir, "four_model_validation.csv"), header, rows)
    write_markdown(os.path.join(args.out_dir, "four_model_validation.md"), header, rows)

    model_names, patient_rows = patient_level_table(model_paths, args.condition, region="WT")
    header2 = ["patient"] + model_names
    write_csv(os.path.join(args.out_dir, f"patient_level_{args.condition}_WT.csv"), header2, patient_rows)
    write_markdown(os.path.join(args.out_dir, f"patient_level_{args.condition}_WT.md"), header2, patient_rows)

    print(f"[compare] wrote tables to {args.out_dir}/")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
