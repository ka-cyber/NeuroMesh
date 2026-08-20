"""
evaluate_frozen_test.py

Evaluates the FROZEN checkpoint (per PROTOCOL_FREEZE_v1.md) on the untouched
test split. Does not train, does not modify the model, does not touch
train/val data. Reconstructs full per-patient volumes (not per-slice
averages) and computes WT/TC/ET Dice, HD95, sensitivity, precision at native
(240x240) resolution -- matching how BraTS itself scores submissions.

Only two experiment groups are run here, per the frozen protocol:
  1. Clean test.
  2. Single-modality-missing (FLAIR, T1, T1ce, T2 -- one at a time).

Nothing else (random dropout sweeps, failure models, baselines) runs from
this script by design -- that's a deliberate, protocol-frozen scope limit,
not an oversight.

Usage:
    python evaluate_frozen_test.py \
        --checkpoint checkpoints/neuromesh_real30.pt \
        --results-out results/FROZEN_TEST_neuromesh_real30.json
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from neuromesh.training.train_real import build_test_dataset, build_datasets, build_model, _BaselineForwardWrapper, maybe_resize
from neuromesh.data.brats_loader import MODALITY_ORDER
from neuromesh.evaluation.metrics import compute_patient_region_metrics, apply_modality_dropout


def reconstruct_patient_volumes(model, test_ds, device, image_size, modality_idx_to_drop=None):
    """
    Groups the flat H5SliceBraTSDataset by patient (volume id), runs every
    slice through the model (at training resolution), upsamples predictions
    back to native 240x240 (nearest-neighbor -- these are discrete labels),
    and returns {volume_id: (pred_volume[155,240,240], target_volume[155,240,240])}.

    modality_idx_to_drop: if set, zero that channel on every slice before
    the forward pass (single-modality-missing condition).

    `model` must already implement forward(x, h_state) -> (logits, h_state,
    mask) -- NeuroMeshUNet does directly; baselines need to be passed in
    wrapped in _BaselineForwardWrapper first (caller's responsibility, same
    pattern as utils.evaluation.evaluate_modality_dropout).
    """
    # Group row indices by volume, sorted by slice index for correct z-order.
    by_volume = defaultdict(list)
    for i, row in enumerate(test_ds.rows):
        by_volume[int(row["volume"])].append((int(row["slice"]), i))
    for v in by_volume:
        by_volume[v].sort()  # sort by slice index

    model.eval()
    out = {}
    with torch.no_grad():
        for volume_id, slice_idx_pairs in by_volume.items():
            indices = [i for _, i in slice_idx_pairs]
            xs, ys = [], []
            for i in indices:
                x, y = test_ds[i]  # x: [4,H,W] native 240x240, y: [H,W] native
                xs.append(x)
                ys.append(y)
            x_native = torch.stack(xs)  # [155,4,240,240]
            y_native = torch.stack(ys)  # [155,240,240]

            if modality_idx_to_drop is not None:
                x_native = x_native.clone()
                x_native[:, modality_idx_to_drop, :, :] = 0.0

            x_resized = F.interpolate(x_native, size=(image_size, image_size),
                                       mode="bilinear", align_corners=False).to(device)

            h_state = None
            pred_slices = []
            batch_size = 8
            for start in range(0, x_resized.size(0), batch_size):
                xb = x_resized[start:start + batch_size]
                if h_state is not None and h_state.size(0) != xb.size(0):
                    h_state = None
                logits, h_state, _ = model(xb, h_state)
                pred_small = logits.argmax(dim=1)  # [b, image_size, image_size]
                pred_slices.append(pred_small.cpu())
            pred_small_vol = torch.cat(pred_slices, dim=0)  # [155, image_size, image_size]

            # Nearest-neighbor upsample predictions back to native resolution
            # for metric computation against the full-fidelity ground truth.
            pred_native = F.interpolate(
                pred_small_vol.unsqueeze(1).float(), size=(240, 240), mode="nearest"
            ).squeeze(1).long()

            out[volume_id] = (pred_native.numpy(), y_native.numpy())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--results-out", required=True)
    p.add_argument("--split", choices=["test", "val"], default="test",
                    help="Which held-out split to evaluate. 'test' touches the frozen test set -- "
                         "only use for an official, protocol-sanctioned evaluation. 'val' is safe to "
                         "run as often as needed during model development/checkpoint comparison.")
    cli_args = p.parse_args()

    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    device = torch.device(saved_args.device)

    print(f"[evaluate_frozen_test] checkpoint = {cli_args.checkpoint}")
    print(f"[evaluate_frozen_test] model={saved_args.model} image_size={saved_args.image_size} "
          f"base_ch={saved_args.base_ch}")

    if cli_args.split == "test":
        test_ds, test_ids = build_test_dataset(saved_args)
    else:
        _train_ds, val_ds, _n_train_p, _n_val_p = build_datasets(saved_args)
        test_ds = val_ds
        test_ids = sorted(set(int(r["volume"]) for r in val_ds.rows))
    print(f"[evaluate_frozen_test] SPLIT = {cli_args.split.upper()}"
          + ("  (touches frozen test set)" if cli_args.split == "test" else "  (val only, test untouched)"))
    print(f"[evaluate_frozen_test] {cli_args.split} patients (n={len(test_ids)}): {test_ids}")
    print(f"[evaluate_frozen_test] test slices: {len(test_ds)}")

    model = build_model(saved_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Same wrapper needed here as elsewhere in this repo: NeuroMeshUNet
    # natively implements forward(x, h_state) -> (logits, h_state, mask);
    # baselines only implement forward(x) -> logits.
    eval_model = model if saved_args.model == "neuromesh" else _BaselineForwardWrapper(model)

    conditions = [{"name": "clean", "modality_idx": None}]
    conditions += [{"name": f"{name}_missing", "modality_idx": i} for i, name in enumerate(MODALITY_ORDER)]

    all_results = {}
    for cond in conditions:
        print(f"\n[evaluate_frozen_test] === condition: {cond['name']} ===")
        t0 = time.time()
        volumes = reconstruct_patient_volumes(
            eval_model, test_ds, device, saved_args.image_size,
            modality_idx_to_drop=cond["modality_idx"],
        )
        per_patient = {}
        for volume_id, (pred_vol, target_vol) in volumes.items():
            m = compute_patient_region_metrics(pred_vol, target_vol, spacing=(1.0, 1.0, 1.0))
            per_patient[volume_id] = m
            print(f"  patient {volume_id}: "
                  f"WT_dice={m['WT']['dice']:.4f} TC_dice={m['TC']['dice']:.4f} ET_dice={m['ET']['dice']:.4f}")

        # Mean across the 4 test patients, per region/metric. None values
        # (undefined, e.g. no ET in that patient) are excluded from the mean
        # and the exclusion count is recorded -- not silently treated as 0.
        summary = {}
        for region in ["WT", "TC", "ET"]:
            summary[region] = {}
            for metric in ["dice", "hd95_mm", "sensitivity", "precision"]:
                vals = [per_patient[v][region][metric] for v in per_patient
                        if per_patient[v][region][metric] is not None]
                n_excluded = len(per_patient) - len(vals)
                summary[region][metric] = {
                    "mean": float(np.mean(vals)) if vals else None,
                    "std": float(np.std(vals)) if vals else None,
                    "n_patients": len(vals),
                    "n_excluded_undefined": n_excluded,
                }

        dt = time.time() - t0
        all_results[cond["name"]] = {"per_patient": per_patient, "summary": summary, "wall_clock_s": dt}
        print(f"  [{dt:.0f}s] WT Dice mean={summary['WT']['dice']['mean']:.4f}  "
              f"TC Dice mean={summary['TC']['dice']['mean']:.4f}  "
              f"ET Dice mean={summary['ET']['dice']['mean'] if summary['ET']['dice']['mean'] is not None else float('nan'):.4f}"
              f" (n_excluded={summary['ET']['dice']['n_excluded_undefined']})")

    out = {
        "REAL_DATA": True,
        "SPLIT": cli_args.split,
        "UNTOUCHED_TEST_SET": cli_args.split == "test",
        "protocol_freeze_doc": "PROTOCOL_FREEZE_v1.md",
        "checkpoint": cli_args.checkpoint,
        "checkpoint_sha256": ckpt.get("_sha256_note", "see PROTOCOL_FREEZE_v1.md"),
        "patient_ids": test_ids,
        "n_patients": len(test_ids),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": all_results,
    }
    with open(cli_args.results_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[evaluate_frozen_test] saved -> {cli_args.results_out}")


if __name__ == "__main__":
    main()
