"""
2-panel figure: Von Neumann lattice with constant weights vs random sign weights.

(a) All weights = 1 (uniform)
(b) Weights randomly ±1 (50/50)

Usage:
    python -m utils.plot_lattice_neighborhoods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

OUTPUT_PATH = "experiments/conference/methods/lattice_sign_illustration.png"
LATTICE_DIM = 6


def _draw_lattice(ax, G, title):
    pos = {nd: (nd[1], -nd[0]) for nd in G.nodes()}

    # Color edges by sign
    edge_colors = ["#4a90d9" if d["weight"] > 0 else "#d94a4a"
                   for _, _, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=150,
                           node_color="lightgray", edgecolors="black", linewidths=0.8)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color=edge_colors, width=1.2,
                           connectionstyle="arc3,rad=0.12", arrows=True, arrowsize=6)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def main():
    m = LATTICE_DIM
    np.random.seed(42)

    # Build Von Neumann directed lattice
    G_base = nx.grid_graph([m, m]).to_directed()

    # (a) Constant weights = 1
    G_const = G_base.copy()
    for u, v, d in G_const.edges(data=True):
        d["weight"] = 1.0

    # (b) Random ±1 weights
    G_sign = G_base.copy()
    for u, v, d in G_sign.edges(data=True):
        d["weight"] = 1.0 if np.random.random() > 0.5 else -1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    _draw_lattice(axes[0], G_const, r"(a) Constant weights ($w = 1$)")
    _draw_lattice(axes[1], G_sign, r"(b) Random sign ($w = \pm 1$)")

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color="#4a90d9", lw=2, label="Positive (+1)"),
        Line2D([0], [0], color="#d94a4a", lw=2, label="Negative (−1)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
