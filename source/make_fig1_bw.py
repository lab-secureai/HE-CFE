"""Black-and-white redraw of Fig. 1 (HE-CFE deployment). Run from REVISION_2026-10/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "figures_2026-10-31" / "fig1_bw"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 10})
fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set(xlim=(0, 14), ylim=(-.75, 6.1))
ax.axis("off")


def region(x, y, w, h, label, above=True):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="black", lw=.8, ls=(0, (4, 3))))
    ax.text(x + w / 2, y + h + .12 if above else y - .12, label, ha="center", va="bottom" if above else "top", fontsize=9.5, style="italic")


def node(cx, cy, text, w=2.4, h=1.3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=.12", fc="white", ec="black", lw=1, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", linespacing=1.1, zorder=4)


def arrow(points):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color="black", lw=1.2, solid_capstyle="butt", zorder=2)
    ax.annotate("", xy=points[-1], xytext=points[-2], arrowprops=dict(arrowstyle="-|>,head_width=.22,head_length=.45", color="black", lw=1.2, shrinkA=0, shrinkB=0), zorder=2)


region(.15, .35, 9.1, 5.1, "Client (holds the secret key)")
region(10.35, .35, 3.5, 2.55, "Server (honest-but-curious,\nno secret key)", above=False)
top, low = 4.3, 1.6
node(1.5, top, "Private\nfactual $x$ and\nsearch state")
node(4.7, top, "Generate and\nproject candidate\npopulation", w=2.75)
node(7.9, top, "CKKS encrypt\npopulation")
node(4.7, low, "Decrypt scores\nand update\narchive", w=2.75)
node(12.1, low, "Encrypted\nMLP scoring")
arrow([(2.7, top), (3.325, top)])
arrow([(6.075, top), (6.7, top)])
arrow([(9.1, top), (12.1, top), (12.1, low + .65)])
arrow([(10.9, low), (6.075, low)])
arrow([(4.7, low + .65), (4.7, top - .65)])
ax.text(10.65, top + .1, "ciphertexts", ha="center", va="bottom", fontsize=9.5)
ax.text(7.75, low + .1, "encrypted scores", ha="center", va="bottom", fontsize=9.5)
fig.savefig(OUT / "fig1_deployment_bw.pdf", bbox_inches="tight", pad_inches=.03)
fig.savefig(OUT / "fig1_deployment_bw.png", dpi=600, bbox_inches="tight", pad_inches=.03, facecolor="white")
