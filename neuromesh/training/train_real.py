"""
train_real.py

Real-data counterpart to train.py. train.py is deliberately left alone (the
manuscript's Reproducibility section and the CI workflow both point at it as
the zero-dependency synthetic smoke test) -- this script is the one that
actually trains on real, licensed BraTS volumes, with a patient-level
train/val/test split, and writes results that are explicitly labeled as real
so they never get confused with the synthetic pipeline-check numbers in
results/synthetic_eval_results.json.

Two supported real data sources (see data/brats_loader.py for details on both):

  --dataset brats_h5     The pre-sliced Kaggle mirror
                          (kaggle.com/datasets/awsaf49/brats2020-training-data).
                          Needs --data-root (dir of .h5 files) and
                          --metadata-csv (BraTS20_Training_Metadata.csv).
                          Patient-level split via build_volume_split().

  --dataset brats_nifti  Official NIfTI layout (CBICA portal or a NIfTI-format
                          mirror). Needs --data-root pointing at a directory
                          tree containing per-case folders. Patient-level split
                          via split_nifti_case_dirs().

Supported --model values: neuromesh (default), dropout, ensemble, plain.
Only `neuromesh` uses the full composite loss (task + rewire + sparsity +
diversity); the baselines train with task loss only, since they have no
controller/mask to regularize.

This script does NOT claim any number it produces is a final, publication-
ready result -- that depends on dataset size, split, hyperparameters, and
epochs actually used, all of which are printed and saved alongside the
metrics for exactly that reason. Every run writes full provenance
(git commit, args, dataset sizes, timestamp) into its results JSON.

Example:
    python train_real.py --dataset brats_h5 \\
        --data-root /path/to/h5_files \\
        --metadata-csv /path/to/BraTS20_Training_Metadata.csv \\
        --model neuromesh --epochs 10 --batch-size 8
"""

import argparse
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from neuromesh.data.brats_loader import (
    BraTSDataset,
    H5SliceBraTSDataset,
    build_volume_split,
    split_nifti_case_dirs,
    MODALITY_ORDER,
)
from neuromesh.models.neuromesh import NeuroMeshUNet
from neuromesh.models.baselines import DropoutUNet, EnsembleUNet, _PlainUNet, StaticGatedUNet
from neuromesh.losses.losses import NeuroMeshLoss, task_loss
from neuromesh.evaluation.metrics import evaluate_modality_dropout, apply_modality_dropout


class _ResizeWrapper(Dataset):
    """Downsamples (x, y) pairs to a smaller spatial size purely to make
    training tractable on constrained hardware (e.g. a single CPU core).
    This is a compute-budget accommodation, not a methodological
    recommendation -- report the resolution actually used, and re-run at full
    resolution on real compute (GPU) before treating any number here as final."""

    def __init__(self, base_dataset, size):
        self.base = base_dataset
        self.size = size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = F.interpolate(x.unsqueeze(0), size=(self.size, self.size), mode="bilinear",
                           align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0).float(), size=(self.size, self.size),
                           mode="nearest").squeeze(0).squeeze(0).long()
        return x, y


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["brats_h5", "brats_nifti"], required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--metadata-csv", default=None, help="Required for --dataset brats_h5")

    p.add_argument("--model", choices=["neuromesh", "dropout", "ensemble", "plain", "static_gate"], default="neuromesh")
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-classes", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--modality-dropout-prob", type=float, default=0.15,
                    help="Applied during training only, as augmentation.")

    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--tumor-only", action="store_true", help="brats_h5 only: keep only slices with visible tumor")
    p.add_argument("--min-tumor-pixels", type=int, default=0, help="brats_h5 only")
    p.add_argument("--max-train-patients", type=int, default=None,
                    help="Cap the number of *training* patients (post-split) -- useful for a fast, "
                         "honest small-N run before committing to a full multi-hour job.")
    p.add_argument("--image-size", type=int, default=None,
                    help="Downsample slices to this size (square) before training. A compute-budget "
                         "accommodation for constrained hardware, not a methodological choice -- "
                         "report whatever value is used, and treat results at reduced resolution as "
                         "preliminary until reproduced at native (240x240) resolution.")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--run-name", default=None, help="Defaults to '<model>_<dataset>'")
    p.add_argument("--max-seconds", type=float, default=None,
                    help="Wall-clock budget for THIS invocation's training loop. On constrained "
                         "hardware where one call can't finish all epochs, stop cleanly at this "
                         "budget, checkpoint (model/optimizer/step count), and exit 0 -- re-running "
                         "the identical command resumes from that checkpoint rather than restarting. "
                         "Final eval + results JSON only get written once training actually completes.")
    p.add_argument("--init-from-checkpoint", default=None,
                    help="Load model weights (only) from a prior run's FINAL checkpoint (the "
                         "no-optimizer-state .pt saved on completion, not the _inprogress.pt) before "
                         "training starts. For continuing training beyond a completed run's original "
                         "--epochs target under a NEW run-name/epoch-count-so-far. Optimizer state is "
                         "NOT preserved across this boundary (the final checkpoint doesn't store it) --"
                         " AdamW momentum/variance restart from zero. This is a disclosed, minor "
                         "deviation from perfectly continuous training, not full resumption; report it "
                         "as such rather than claiming uninterrupted training across the boundary.")
    return p.parse_args()


def build_datasets(args):
    if args.dataset == "brats_h5":
        if args.metadata_csv is None:
            raise ValueError("--metadata-csv is required for --dataset brats_h5")
        train_ids, val_ids, _test_ids = build_volume_split(
            args.metadata_csv, val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, seed=args.split_seed,
        )
        if args.max_train_patients is not None:
            train_ids = set(sorted(train_ids)[: args.max_train_patients])

        train_ds = H5SliceBraTSDataset(
            args.metadata_csv, args.data_root, volume_ids=train_ids,
            tumor_only=args.tumor_only, min_tumor_pixels=args.min_tumor_pixels,
            modality_dropout_prob=args.modality_dropout_prob,
        )
        val_ds = H5SliceBraTSDataset(
            args.metadata_csv, args.data_root, volume_ids=val_ids,
            tumor_only=False, min_tumor_pixels=0, modality_dropout_prob=0.0,
        )
        n_train_patients, n_val_patients = len(train_ids), len(val_ids)

    else:  # brats_nifti
        train_dirs, val_dirs, _test_dirs = split_nifti_case_dirs(
            args.data_root, val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, seed=args.split_seed,
        )
        if args.max_train_patients is not None:
            train_dirs = train_dirs[: args.max_train_patients]

        train_ds = BraTSDataset(args.data_root, case_dirs=train_dirs,
                                 modality_dropout_prob=args.modality_dropout_prob)
        val_ds = BraTSDataset(args.data_root, case_dirs=val_dirs,
                               modality_dropout_prob=0.0)
        n_train_patients, n_val_patients = len(train_dirs), len(val_dirs)

    return train_ds, val_ds, n_train_patients, n_val_patients


def build_test_dataset(args):
    """Builds ONLY the test set -- deliberately separate from build_datasets()
    (used by the training loop) so there is no code path in the training
    script that can touch test data, by construction, not just by convention.
    Only ever called from a dedicated evaluation script, never from main()."""
    if args.dataset == "brats_h5":
        if args.metadata_csv is None:
            raise ValueError("--metadata-csv is required for --dataset brats_h5")
        _train_ids, _val_ids, test_ids = build_volume_split(
            args.metadata_csv, val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, seed=args.split_seed,
        )
        test_ds = H5SliceBraTSDataset(
            args.metadata_csv, args.data_root, volume_ids=test_ids,
            tumor_only=False, min_tumor_pixels=0, modality_dropout_prob=0.0,
        )
        return test_ds, sorted(test_ids)
    else:  # brats_nifti
        _train_dirs, _val_dirs, test_dirs = split_nifti_case_dirs(
            args.data_root, val_fraction=args.val_fraction,
            test_fraction=args.test_fraction, seed=args.split_seed,
        )
        test_ds = BraTSDataset(args.data_root, case_dirs=test_dirs, modality_dropout_prob=0.0)
        return test_ds, sorted(test_dirs)


def maybe_resize(ds, args):
    if args.image_size is not None:
        return _ResizeWrapper(ds, args.image_size)
    return ds


def build_model(args, device):
    if args.model == "neuromesh":
        return NeuroMeshUNet(in_channels=4, num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    elif args.model == "dropout":
        return DropoutUNet(in_channels=4, num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    elif args.model == "ensemble":
        return EnsembleUNet(in_channels=4, num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    elif args.model == "plain":
        return _PlainUNet(in_channels=4, num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    elif args.model == "static_gate":
        return StaticGatedUNet(in_channels=4, num_classes=args.num_classes, base_ch=args.base_ch).to(device)
    raise ValueError(args.model)


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def train_step(model, x, y, criterion, optimizer, device, args, h_state):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    if args.model == "neuromesh":
        # Same batch-size-mismatch guard as utils/evaluation.py -- real data's
        # last batch of an epoch is often smaller than the rest.
        if h_state is not None and h_state.size(0) != x.size(0):
            h_state = None
        logits_clean, h_state, mask = model(x, h_state.detach() if h_state is not None else None)
        x_corrupted = apply_modality_dropout(x, drop_prob=0.15)
        logits_corrupted, _, mask2 = model(x_corrupted, h_state.detach())
        loss, components = criterion(logits_clean, y, logits_corrupted=logits_corrupted, masks=[mask, mask2])
    else:  # dropout, ensemble, plain
        logits = model(x)
        loss = task_loss(logits, y, args.num_classes)
        components = {"task": loss}

    loss.backward()
    optimizer.step()
    return loss.item(), h_state


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_name = args.run_name or f"{args.model}_{args.dataset}"

    print(f"[train_real] run = {run_name}")
    print(f"[train_real] device = {device}  (no GPU detected -> CPU; expect real training to be slow "
          f"at full BraTS scale)" if device.type == "cpu" else f"[train_real] device = {device}")

    train_ds, val_ds, n_train_p, n_val_p = build_datasets(args)
    print(f"[train_real] patients: train={n_train_p} val={n_val_p}  "
          f"| slices: train={len(train_ds)} val={len(val_ds)}")
    if n_train_p < 20:
        print(f"[train_real] WARNING: only {n_train_p} training patients. This is a small-N run -- "
              f"report it as such, not as a final result, in any manuscript.")
    if args.image_size is not None:
        print(f"[train_real] NOTE: downsampling to {args.image_size}x{args.image_size} "
              f"(native resolution is 240x240) -- a compute accommodation, not a methodological choice.")
    train_ds = maybe_resize(train_ds, args)
    val_ds = maybe_resize(val_ds, args)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    batches_per_epoch = len(train_loader)
    total_steps_target = batches_per_epoch * args.epochs

    model = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = NeuroMeshLoss(num_classes=args.num_classes) if args.model == "neuromesh" else None

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(args.checkpoint_dir, f"{run_name}_inprogress.pt")

    global_step = 0
    loss_history = []
    train_wall_s_so_far = 0.0
    if args.init_from_checkpoint is not None and not os.path.exists(resume_path):
        # Only applies at the very start of THIS run-name's training (not on
        # subsequent resumes of an already-started in-progress checkpoint,
        # which takes priority below).
        init_ckpt = torch.load(args.init_from_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(init_ckpt["model_state_dict"])
        print(f"[train_real] initialized weights from {args.init_from_checkpoint} "
              f"(optimizer state NOT preserved -- AdamW momentum restarts from zero)")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["global_step"]
        loss_history = ckpt["loss_history"]
        train_wall_s_so_far = ckpt.get("train_wall_clock_s", 0.0)
        print(f"[train_real] RESUMING from {resume_path}: global_step={global_step}/{total_steps_target} "
              f"({train_wall_s_so_far:.0f}s of training already logged)")
    else:
        print(f"[train_real] model={args.model} params={n_params:,}  "
              f"target: {args.epochs} epochs x {batches_per_epoch} batches/epoch = {total_steps_target} steps")

    model.train()
    h_state = None
    t0 = time.time()
    train_iter = iter(train_loader)
    while global_step < total_steps_target:
        if args.max_seconds is not None and (time.time() - t0) >= args.max_seconds:
            break
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            epoch_num = global_step // batches_per_epoch
            mean_epoch_loss = sum(loss_history[-batches_per_epoch:]) / batches_per_epoch
            print(f"[epoch {epoch_num}/{args.epochs}] mean_loss={mean_epoch_loss:.4f}")

        loss_val, h_state = train_step(model, x, y, criterion, optimizer, device, args, h_state)
        loss_history.append(loss_val)
        global_step += 1

    train_wall_s_this_call = time.time() - t0
    train_wall_s_total = train_wall_s_so_far + train_wall_s_this_call

    if global_step < total_steps_target:
        # Ran out of wall-clock budget for this call -- checkpoint and exit
        # cleanly (return code 0) so re-running the same command resumes.
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "loss_history": loss_history,
            "train_wall_clock_s": train_wall_s_total,
        }, resume_path)
        print(f"\n[train_real] PAUSED at step {global_step}/{total_steps_target} "
              f"({100*global_step/total_steps_target:.1f}%) -- hit --max-seconds={args.max_seconds} budget.")
        print(f"[train_real] checkpoint saved -> {resume_path}")
        print(f"[train_real] total training time logged so far: {train_wall_s_total:.0f}s")
        print("[train_real] re-run the identical command to resume.")
        return

    print(f"\n[train_real] training complete: {global_step} steps ({args.epochs} epochs) "
          f"in {train_wall_s_total:.0f}s total wall-clock.")
    print(f"\n[train_real] Evaluating modality-dropout robustness on REAL {args.dataset} val set "
          f"({n_val_p} held-out patients)...")
    dropout_configs = [{"name": "clean", "modality_idx": None, "drop_prob": 0.0}]
    dropout_configs += [{"name": f"{name}_missing", "modality_idx": i, "drop_prob": 0.0}
                         for i, name in enumerate(MODALITY_ORDER)]
    dropout_configs += [
        {"name": "random_20pct", "modality_idx": None, "drop_prob": 0.2},
        {"name": "random_50pct", "modality_idx": None, "drop_prob": 0.5},
    ]

    # evaluate_modality_dropout is written against NeuroMeshUNet's
    # forward(x, h_state) -> (logits, h_state, mask) contract, since it threads
    # a hidden state across the batch loop. NeuroMeshUNet matches that
    # directly; the baselines only take forward(x) -> logits, so they need a
    # thin adapter that accepts and passes through the (unused) h_state.
    eval_model = model if args.model == "neuromesh" else _BaselineForwardWrapper(model)
    results = evaluate_modality_dropout(eval_model, val_loader, args.num_classes, device, dropout_configs)

    print("\n%-16s %10s %14s %14s" % ("condition", "mean_dice", "mean_lat(ms)", "max_lat(ms)"))
    for name, r in results.items():
        print("%-16s %10.4f %14.3f %14.3f" % (name, r["mean_dice"], r["mean_latency_ms"], r["max_latency_ms"]))

    ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}.pt")
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, ckpt_path)
    if os.path.exists(resume_path):
        os.remove(resume_path)

    out = {
        "REAL_DATA": True,
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "git_commit": git_commit_hash(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_train_patients": n_train_p,
        "n_val_patients": n_val_p,
        "n_train_slices": len(train_ds),
        "n_val_slices": len(val_ds),
        "epochs": args.epochs,
        "train_wall_clock_s": train_wall_s_total,
        "device": str(device),
        "args": vars(args),
        "train_loss_history": loss_history,
        "eval": results,
        "checkpoint_path": ckpt_path,
    }
    out_path = os.path.join(args.results_dir, f"real_{run_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[train_real] saved checkpoint -> {ckpt_path}")
    print(f"[train_real] saved results   -> {out_path}")
    print(
        f"\nNOTE: trained on {n_train_p} real patients, evaluated on {n_val_p} held-out real "
        f"patients (patient-level split, seed={args.split_seed}). "
        + ("This is a small-N run -- treat as preliminary, not a final reported result."
           if n_train_p < 20 else
           "Report exact split sizes/seed alongside these numbers in the manuscript.")
    )


class _BaselineForwardWrapper(torch.nn.Module):
    """Adapts a baseline's forward(x) -> logits into the
    forward(x, h_state) -> (logits, h_state, mask) contract that
    evaluate_modality_dropout expects (it was written against NeuroMeshUNet
    and threads a hidden state through every call in its loop).  h_state is
    accepted and passed straight through, unused -- the baselines are
    stateless, there's no controller hidden state to carry."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, h_state=None):
        logits = self.model(x)
        return logits, h_state, None

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self


if __name__ == "__main__":
    main()
