"""
figures/fig_b_dropout_panel.py

WHY THIS SCRIPT LOOKS DIFFERENT FROM WHAT WAS REQUESTED
---------------------------------------------------------
The original ask was for a panel that *shows* "100% loss of the FLAIR channel
still yields highly precise, stable ground-truth tumor segmentation masks."
That's a specific empirical claim, and no real BraTS training/validation has
happened in this environment -- there's no licensed data here, and this
sandbox has no network path to the BraTS portals. Pre-deciding that the result
will look "highly precise" and building a figure around that conclusion would
mean putting a fabricated finding into a manuscript, which isn't something to
do even for a fast rebuild.

What this script does instead: it trains NeuroMeshUNet briefly on the
dependency-free MockBraTSDataset, then genuinely runs it on clean vs.
FLAIR-dropped vs. T1ce-dropped inputs and plots WHATEVER the model actually
predicts, with the real measured per-sample Dice printed on each panel. On a
few epochs of synthetic data this will look mediocre -- that's expected and
correct. Once you swap in real BraTS data (see data/brats_loader.py) and train
properly, rerun this exact script to get your real Figure B.

Run:
    python figures/fig_b_dropout_panel.py
Produces:
    figures/fig_b_dropout_panel.png
"""

import sys
import os

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromesh.data.brats_loader import MockBraTSDataset, build_dataloader
from neuromesh.models.neuromesh import NeuroMeshUNet
from neuromesh.losses.losses import NeuroMeshLoss
from neuromesh.evaluation.metrics import apply_modality_dropout, dice_per_class


MODALITY_NAMES = ["T1", "T1ce", "T2", "FLAIR"]


def quick_train(model, loader, device, epochs=5, lr=1e-3, num_classes=4):
    criterion = NeuroMeshLoss(num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for ep in range(epochs):
        h_state = None
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, h_state, mask = model(x, h_state.detach() if h_state is not None else None)
            x_corr = apply_modality_dropout(x, drop_prob=0.15)
            logits_corr, _, mask2 = model(x_corr, h_state.detach())
            loss, _ = criterion(logits, y, logits_corrupted=logits_corr, masks=[mask, mask2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  [quick_train] epoch {ep + 1}/{epochs} done")
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    train_ds = MockBraTSDataset(n_samples=32, image_size=(96, 96), num_classes=4, seed=10)
    train_loader = build_dataloader(train_ds, batch_size=4, shuffle=True)

    model = NeuroMeshUNet(in_channels=4, num_classes=4, base_ch=16).to(device)
    print("[fig_b] quick-training on synthetic data (NOT a substitute for real BraTS training)...")
    quick_train(model, train_loader, device, epochs=5)

    # one held-out synthetic example
    val_ds = MockBraTSDataset(n_samples=1, image_size=(96, 96), num_classes=4, seed=999)
    x, y = val_ds[0]
    x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

    conditions = [
        ("clean", None),
        ("FLAIR dropped (100%)", 3),
        ("T1ce dropped (100%)", 1),
    ]

    fig, axes = plt.subplots(len(conditions), 4, figsize=(13, 3.4 * len(conditions)))

    for row, (label, modality_idx) in enumerate(conditions):
        x_in = apply_modality_dropout(x, modality_idx=modality_idx)
        with torch.no_grad():
            logits, _, _ = model(x_in)
            pred = logits.argmax(dim=1)
            dices = dice_per_class(logits, y, num_classes=4)

        # show all 4 input channels stacked as a single montage strip, the GT, and the prediction
        montage = torch.cat([x_in[0, c] for c in range(4)], dim=1).cpu()
        axes[row, 0].imshow(montage, cmap="gray")
        axes[row, 0].set_title(f"{label}\ninput channels: {' | '.join(MODALITY_NAMES)}", fontsize=8)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(y[0].cpu(), cmap="viridis", vmin=0, vmax=3)
        axes[row, 1].set_title("ground truth (synthetic)", fontsize=8)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred[0].cpu(), cmap="viridis", vmin=0, vmax=3)
        mean_dice = sum(dices) / len(dices)
        axes[row, 2].set_title(f"prediction\nmean Dice = {mean_dice:.3f} (measured)", fontsize=8)
        axes[row, 2].axis("off")

        diff = (pred[0].cpu() != y[0].cpu()).float()
        axes[row, 3].imshow(diff, cmap="Reds", vmin=0, vmax=1)
        axes[row, 3].set_title("error map (red = mismatch)", fontsize=8)
        axes[row, 3].axis("off")

    fig.suptitle(
        "Figure B -- Modality-dropout panel on SYNTHETIC data, lightly trained model\n"
        "(illustrates the pipeline only -- rerun on real, well-trained BraTS models before any manuscript claim)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_b_dropout_panel.png", dpi=200, bbox_inches="tight")
    print("Saved figures/fig_b_dropout_panel.png")


if __name__ == "__main__":
    main()
