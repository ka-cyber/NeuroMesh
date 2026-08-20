"""
train.py

SYNTHETIC-DATA SMOKE TEST ONLY. This does not train on real BraTS data and
its Dice/latency numbers carry no clinical or scientific meaning -- it
exists purely to verify the pipeline (data loading -> model -> loss ->
evaluation) runs end to end without requiring any real, licensed dataset.

For the actual real-data pipeline used to produce every number in README.md
and results/, use the `neuromesh` CLI instead:

    neuromesh train --dataset brats_h5 --data-root <path> --metadata-csv <path> ...

See docs/dataset.md for how to obtain real data, and `neuromesh reproduce`
for the exact commands used to produce this repository's reported results.

Run this file:
    python train.py --epochs 3 --batch-size 4
"""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from neuromesh.data import MockBraTSDataset
from neuromesh.models import NeuroMeshUNet
from neuromesh.losses import NeuroMeshLoss
from neuromesh.evaluation import evaluate_modality_dropout, apply_modality_dropout


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train", type=int, default=32)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[NeuroMesh] device = {device}")

    train_ds = MockBraTSDataset(args.n_train, (args.image_size, args.image_size), args.num_classes, seed=1)
    val_ds = MockBraTSDataset(args.n_val, (args.image_size, args.image_size), args.num_classes, seed=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = NeuroMeshUNet(in_channels=4, num_classes=args.num_classes, base_ch=16).to(device)
    criterion = NeuroMeshLoss(num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[NeuroMesh] model has {n_params:,} parameters")

    components = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        h_state = None
        epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits_clean, h_state, mask = model(x, h_state.detach() if h_state is not None else None)

            x_corrupted = apply_modality_dropout(x, drop_prob=0.15)
            logits_corrupted, _, mask2 = model(x_corrupted, h_state.detach())

            loss, components = criterion(logits_clean, y, logits_corrupted=logits_corrupted, masks=[mask, mask2])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        print(
            f"[epoch {epoch}/{args.epochs}] mean_loss={mean_loss:.4f} "
            f"(task={components['task'].item():.4f}, "
            f"rewire={components.get('rewire', torch.tensor(0.0)).item():.4f}, "
            f"sparsity={components.get('sparsity', torch.tensor(0.0)).item():.4f}, "
            f"diversity={components.get('diversity', torch.tensor(0.0)).item():.4f})"
        )

    print("\n[NeuroMesh] Evaluating modality-dropout robustness on SYNTHETIC val set...")
    dropout_configs = [
        {"name": "clean", "modality_idx": None, "drop_prob": 0.0},
        {"name": "flair_dropped", "modality_idx": 3, "drop_prob": 0.0},
        {"name": "t1ce_dropped", "modality_idx": 1, "drop_prob": 0.0},
        {"name": "random_20pct", "modality_idx": None, "drop_prob": 0.2},
        {"name": "random_50pct", "modality_idx": None, "drop_prob": 0.5},
    ]
    results = evaluate_modality_dropout(model, val_loader, args.num_classes, device, dropout_configs)

    print("\n%-16s %10s %14s %14s" % ("condition", "mean_dice", "mean_lat(ms)", "max_lat(ms)"))
    for name, r in results.items():
        print("%-16s %10.4f %14.3f %14.3f" % (name, r["mean_dice"], r["mean_latency_ms"], r["max_latency_ms"]))

    print(
        "\nNOTE: these numbers come from a lightly-trained model on SYNTHETIC mock\n"
        "data. They verify the pipeline runs end-to-end; they are NOT a validated\n"
        "clinical result and must not be reported as one in a manuscript."
    )

    with open("synthetic_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
