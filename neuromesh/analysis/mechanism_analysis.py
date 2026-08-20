"""
mechanism_analysis.py

Investigates WHAT the NeuroMesh controller actually does differently across
conditions, on the exact same 4 frozen test patients used in
evaluate_frozen_test.py. Does not retrain or modify the checkpoint -- uses
the return_debug=True instrumentation added to NeuroMeshUNet.forward (purely
additive, verified not to change any computed value -- full test suite
still 26/26 passing after adding it).

Two passes, kept deliberately separate rather than interleaved (simpler,
less bug-prone than fishing a specific slice out of a batched loop):

  PASS 1 (aggregate, all 155 slices/patient, batched): for each
  (patient, condition), the controller's failure_signal (how many of the 256
  bottleneck channels the model itself flags as numerically dead -- a hard,
  automatic signal, not a soft/learned one), gate statistics (mean, std,
  fraction suppressed below 0.1), and mask density (mean over the full
  256x256 edge-activation matrix). Also accumulates the mean mask matrix.

  PASS 2 (one representative slice/patient, single-sample forward, h_state
  reset to None): the slice with the most tumor pixels in the clean
  ground truth, re-run standalone under each condition for a clean,
  history-independent snapshot -- input, prediction, single-slice mask,
  single-slice gate. This is a controlled comparison, not a claim about
  what the controller does mid-sequence during the actual batched eval.

Usage:
    python mechanism_analysis.py --checkpoint checkpoints/neuromesh_real30.pt \
        --out-dir mechanism_out
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from neuromesh.training.train_real import build_test_dataset, build_datasets, build_model
from neuromesh.data.brats_loader import MODALITY_ORDER


def group_by_volume(test_ds):
    by_volume = defaultdict(list)
    for i, row in enumerate(test_ds.rows):
        by_volume[int(row["volume"])].append((int(row["slice"]), i))
    for v in by_volume:
        by_volume[v].sort()
    return by_volume


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--condition", default=None,
                    help="Run PASS 1 for just this one condition (chunking for time-limited "
                         "environments). Appends/merges into the existing summary/mean_masks "
                         "files rather than overwriting. Omit to run PASS 2 (representative "
                         "slices, fast) instead -- run that only after all conditions are done.")
    p.add_argument("--split", choices=["test", "val"], default="test",
                    help="'test' touches the frozen test set -- only for an official, "
                         "protocol-sanctioned evaluation. 'val' is safe to run repeatedly during "
                         "model development.")
    cli_args = p.parse_args()
    os.makedirs(cli_args.out_dir, exist_ok=True)

    ckpt = torch.load(cli_args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])
    device = torch.device(saved_args.device)
    assert saved_args.model == "neuromesh", "mechanism analysis only applies to the neuromesh controller"

    test_ds, test_ids = build_test_dataset(saved_args) if cli_args.split == "test" else (None, None)
    if cli_args.split == "val":
        _train_ds, val_ds, _n_train_p, _n_val_p = build_datasets(saved_args)
        test_ds = val_ds
        test_ids = sorted(set(int(r["volume"]) for r in val_ds.rows))
    by_volume = group_by_volume(test_ds)
    print(f"[mechanism] SPLIT = {cli_args.split.upper()}"
          + ("  (touches frozen test set)" if cli_args.split == "test" else "  (val only, test untouched)"))
    print(f"[mechanism] {cli_args.split} patients: {test_ids}")

    model = build_model(saved_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    conditions = [{"name": "clean", "modality_idx": None}]
    conditions += [{"name": f"{name}_missing", "modality_idx": i} for i, name in enumerate(MODALITY_ORDER)]

    C = saved_args.base_ch * 16  # bottleneck channel count (controller feature_dim)
    print(f"[mechanism] bottleneck channel count = {C}")

    summary_path = os.path.join(cli_args.out_dir, "mechanism_summary.json")
    masks_path = os.path.join(cli_args.out_dir, "mean_masks.npz")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    mean_masks = dict(np.load(masks_path)) if os.path.exists(masks_path) else {}

    if cli_args.condition is not None:
        conditions = [c for c in conditions if c["name"] == cli_args.condition]
        assert len(conditions) == 1, f"unknown condition {cli_args.condition}"

        # ---------- PASS 1 for this one condition ----------
        with torch.no_grad():
            for cond in conditions:
                print(f"\n[mechanism] === PASS 1, condition: {cond['name']} ===")
                for volume_id, pairs in by_volume.items():
                    indices = [i for _, i in pairs]
                    xs = torch.stack([test_ds[i][0] for i in indices]).to(device)
                    if cond["modality_idx"] is not None:
                        xs = xs.clone()
                        xs[:, cond["modality_idx"], :, :] = 0.0

                    h_state = None
                    dead_counts, gate_means, gate_low_frac, mask_densities = [], [], [], []
                    mask_sum = torch.zeros(C, C)
                    bs = 8
                    for start in range(0, xs.size(0), bs):
                        xb = xs[start:start + bs]
                        if h_state is not None and h_state.size(0) != xb.size(0):
                            h_state = None
                        _logits, h_state, mask, debug = model(xb, h_state, return_debug=True)
                        dead_counts.append(debug["failure_signal"].sum(dim=1).cpu())
                        gate_means.append(debug["gate"].mean(dim=1).cpu())
                        gate_low_frac.append((debug["gate"] < 0.1).float().mean(dim=1).cpu())
                        mask_densities.append(mask.mean(dim=(1, 2)).cpu())
                        mask_sum += mask.sum(dim=0).cpu()

                    dead_counts = torch.cat(dead_counts)
                    gate_means = torch.cat(gate_means)
                    gate_low_frac = torch.cat(gate_low_frac)
                    mask_densities = torch.cat(mask_densities)
                    mean_mask = (mask_sum / len(indices)).numpy()

                    key = f"{volume_id}|{cond['name']}"
                    summary[key] = {
                        "patient": volume_id,
                        "condition": cond["name"],
                        "mean_dead_channels": float(dead_counts.float().mean()),
                        "mean_dead_channels_pct": float(dead_counts.float().mean() / C * 100),
                        "mean_gate": float(gate_means.mean()),
                        "mean_frac_gate_below_0.1": float(gate_low_frac.mean()),
                        "mean_mask_density": float(mask_densities.mean()),
                    }
                    mean_masks[key.replace("|", "__")] = mean_mask
                    print(f"  patient {volume_id}: dead_ch={summary[key]['mean_dead_channels']:.1f}/{C} "
                          f"({summary[key]['mean_dead_channels_pct']:.1f}%)  "
                          f"mean_gate={summary[key]['mean_gate']:.4f}  "
                          f"frac_gate<0.1={summary[key]['mean_frac_gate_below_0.1']:.4f}  "
                          f"mask_density={summary[key]['mean_mask_density']:.4f}")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        np.savez(masks_path, **mean_masks)
        print(f"\n[mechanism] merged -> {summary_path}, {masks_path}")
        return

    # ---------- PASS 2: one representative slice/patient, standalone forward ----------
    print("\n[mechanism] === PASS 2: representative-slice snapshots ===")
    rep_slice_idx = {}
    for volume_id, pairs in by_volume.items():
        best_i, best_count = None, -1
        for _, i in pairs:
            _, y = test_ds[i]
            count = int((y > 0).sum())
            if count > best_count:
                best_count, best_i = count, i
        rep_slice_idx[volume_id] = best_i
        print(f"  patient {volume_id}: representative slice index (dataset-local) = {best_i}, "
              f"tumor pixels = {best_count}")

    rep_arrays = {}
    with torch.no_grad():
        for volume_id, idx in rep_slice_idx.items():
            x0, y0 = test_ds[idx]
            rep_arrays[f"{volume_id}__input_clean"] = x0.numpy()
            rep_arrays[f"{volume_id}__target"] = y0.numpy()
            for cond in conditions:
                x = x0.clone().unsqueeze(0)
                if cond["modality_idx"] is not None:
                    x[:, cond["modality_idx"], :, :] = 0.0
                logits, _h, mask, debug = model(x, None, return_debug=True)
                pred_small = logits.argmax(dim=1)
                pred_native = F.interpolate(pred_small.unsqueeze(1).float(), size=(240, 240),
                                             mode="nearest").squeeze().long().numpy()
                rep_arrays[f"{volume_id}__{cond['name']}__pred"] = pred_native
                rep_arrays[f"{volume_id}__{cond['name']}__mask"] = mask[0].numpy()
                rep_arrays[f"{volume_id}__{cond['name']}__gate"] = debug["gate"][0].numpy()

    with open(os.path.join(cli_args.out_dir, "mechanism_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(cli_args.out_dir, "mean_masks.npz"),
             **{k.replace("|", "__"): v for k, v in mean_masks.items()})
    np.savez(os.path.join(cli_args.out_dir, "rep_slices.npz"), **rep_arrays)
    print(f"\n[mechanism] saved -> {cli_args.out_dir}/mechanism_summary.json, mean_masks.npz, rep_slices.npz")


if __name__ == "__main__":
    main()
