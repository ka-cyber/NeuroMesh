"""
figures/fig_c_latency_benchmark.py

Runs utils.evaluation.benchmark_recovery_latency on an actual NeuroMeshUNet
instance and plots MEASURED forward-pass latency vs. simulated fault count.

This produces real numbers from whatever machine you run it on -- CPU here,
GPU if you have one available. It intentionally does NOT reuse the draft
manuscript's "3.2 ms" figure, because that number was never independently
verified in this rebuild. If you want a "critical medical window" threshold
line on the plot, set --budget-ms to whatever your actual deployment
requirement is and justify it separately in the manuscript text.

Run:
    python figures/fig_c_latency_benchmark.py --budget-ms 50
Produces:
    figures/fig_c_latency_benchmark.png
    figures/fig_c_latency_data.json
"""

import argparse
import json
import sys
import os

import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromesh.models.neuromesh import NeuroMeshUNet
from neuromesh.data.brats_loader import MockBraTSDataset, build_dataloader
from neuromesh.evaluation.metrics import benchmark_recovery_latency


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-ms", type=float, default=None,
                    help="Optional latency budget line to draw on the plot (set to your own real requirement).")
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model = NeuroMeshUNet(in_channels=4, num_classes=4, base_ch=16).to(device)
    model.eval()

    ds = MockBraTSDataset(n_samples=args.batch_size, image_size=(args.image_size, args.image_size))
    loader = build_dataloader(ds, batch_size=args.batch_size, shuffle=False)
    sample_batch = next(iter(loader))

    fault_counts = [0, 1, 2, 3, 5, 8, 12, 16, 20]
    print(f"[fig_c] benchmarking on device={device}, image_size={args.image_size}, batch={args.batch_size}")
    results = benchmark_recovery_latency(model, sample_batch, fault_counts, device, n_repeats=15)

    for fc, r in results.items():
        print(f"  fault_count={fc:>3d}  mean={r['mean_ms']:.3f} ms  std={r['std_ms']:.3f} ms")

    os.makedirs("figures", exist_ok=True)
    with open("figures/fig_c_latency_data.json", "w") as f:
        json.dump(results, f, indent=2)

    xs = list(results.keys())
    means = [results[x]["mean_ms"] for x in xs]
    stds = [results[x]["std_ms"] for x in xs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, means, yerr=stds, marker="o", color="#2980b9", capsize=3, label="measured forward latency")
    if args.budget_ms is not None:
        ax.axhline(args.budget_ms, color="#c0392b", linestyle="--", label=f"latency budget ({args.budget_ms:.1f} ms)")
    ax.set_xlabel("simulated fault count (channels zeroed per forward pass)")
    ax.set_ylabel("latency (ms)")
    ax.set_title(f"Figure C -- Measured recovery latency vs. fault count\n(device={device}, this is a real benchmark, not an assumed constant)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/fig_c_latency_benchmark.png", dpi=200, bbox_inches="tight")
    print("Saved figures/fig_c_latency_benchmark.png")


if __name__ == "__main__":
    main()
