"""
figures/fig_a_cortical_analogy.py

Generates a CONCEPTUAL schematic pairing an edge-activation mask (left) with a
stylized diagram of axonal sprouting / synaptic rerouting after injury (right).

IMPORTANT FRAMING NOTE FOR YOUR MANUSCRIPT
--------------------------------------------
This figure is an *illustrative analogy*, not evidence that NeuroMesh's mask
dynamics recapitulate real cortical plasticity mechanisms. If you use it in a
paper, it should be captioned as a conceptual motivation figure (e.g. "Figure 1:
Conceptual analogy between..."), not placed in a Results section or cited as
biological validation. Real claims about correspondence to axonal sprouting
would need actual neuroscience literature citations and, ideally, a
neuroscientist co-author's review -- an MBBS candidate co-author's judgment
call is a reasonable start, but "looks similar to" is not the same claim as
"is evidence of."

Run:
    python figures/fig_a_cortical_analogy.py
Produces:
    figures/fig_a_cortical_analogy.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

rng = np.random.default_rng(0)

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# ---- Left panel: an actual (small, illustrative) edge-activation mask ----
n = 10
mask = rng.random((n, n))
mask[mask < 0.55] = 0  # sparsify, matching the ~30-50% active-edge range in utils/loss.py
ax = axes[0]
im = ax.imshow(mask, cmap="viridis", vmin=0, vmax=1)
ax.set_title("NeuroMesh edge-activation mask $m^{(\\ell)}$\n(engineering artifact: learned, per-layer)", fontsize=10)
ax.set_xlabel("target node")
ax.set_ylabel("source node")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="activation strength")

# ---- Right panel: stylized (non-clinical, illustrative) sprouting diagram ----
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("Illustrative analogy:\naxonal sprouting / rerouting after injury\n(schematic -- not a real histology image)", fontsize=10)

# damaged pathway (dashed, red X)
ax2.plot([1, 5], [7, 5], color="#c0392b", lw=2, linestyle="--")
ax2.plot([4.7, 5.3], [4.7, 5.3], color="#c0392b", lw=2)
ax2.plot([4.7, 5.3], [5.3, 4.7], color="#c0392b", lw=2)
ax2.text(2.5, 7.5, "original pathway\n(disrupted)", fontsize=8, color="#c0392b", ha="center")

# sprouted alternate pathways (solid, green, curved)
for (x0, y0, x1, y1, curve) in [(1, 7, 9, 3, 0.6), (1, 7, 9, 3, -0.9), (1, 7, 9, 3, 1.6)]:
    xs = np.linspace(x0, x1, 50)
    ys = np.linspace(y0, y1, 50) + curve * np.sin(np.linspace(0, np.pi, 50))
    ax2.plot(xs, ys, color="#27ae60", lw=1.8, alpha=0.85)
ax2.text(7.5, 2.2, "alternate\npathways", fontsize=8, color="#27ae60", ha="center")

ax2.scatter([1], [7], s=120, color="#2c3e50", zorder=5)
ax2.scatter([9], [3], s=120, color="#2c3e50", zorder=5)
ax2.text(1, 7.7, "source region", fontsize=8, ha="center")
ax2.text(9, 2.3, "target region", fontsize=8, ha="center")

legend_elems = [
    Line2D([0], [0], color="#c0392b", lw=2, ls="--", label="lost connectivity"),
    Line2D([0], [0], color="#27ae60", lw=2, label="rerouted connectivity"),
]
ax2.legend(handles=legend_elems, loc="lower left", fontsize=8, frameon=False)

fig.suptitle("Figure A -- Conceptual analogy only: engineered rewiring vs. biological rerouting",
             fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig("figures/fig_a_cortical_analogy.png", dpi=200, bbox_inches="tight")
print("Saved figures/fig_a_cortical_analogy.png")
