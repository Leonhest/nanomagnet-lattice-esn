"""
Multi-panel figure showing neighborhood shells 1–10 on a small lattice.

Each panel highlights the neighbors of a central node for that shell count,
showing how connectivity grows with each additional distance shell.

Usage:
    python -m visualization.plot_neighborhood_shells
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matrix import Matrix


OUTPUT_PATH = "experiments/neighborhood_sizes/neighborhood_shells.png"
LATTICE_DIM = 9
SHELLS = list(range(1, 11))


class _DummyMatrix(Matrix):
    def __init__(self):
        self.W_res_args = {"type": "constant"}


def main():
    m = _DummyMatrix()
    center = (LATTICE_DIM // 2, LATTICE_DIM // 2)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10))

    for ax, n_shells in zip(axes.flat, SHELLS):
        G = m.tetragonal([LATTICE_DIM, LATTICE_DIM], neighborhood=n_shells)
        neighbors = set(G.neighbors(center))
        offsets = Matrix._shell_offsets(n_shells)

        pos = {nd: (nd[1], -nd[0]) for nd in G.nodes()}

        # Color nodes: center = red, neighbors = blue, others = light gray
        node_colors = []
        for nd in G.nodes():
            if nd == center:
                node_colors.append("#d94a4a")
            elif nd in neighbors:
                node_colors.append("#4a90d9")
            else:
                node_colors.append("#e0e0e0")

        # Only draw edges connected to center
        center_edges = [(u, v) for u, v in G.edges() if u == center or v == center]
        other_edges = [(u, v) for u, v in G.edges() if u != center and v != center]

        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=other_edges,
                               edge_color="#e0e0e0", width=0.3)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=center_edges,
                               edge_color="#4a90d9", width=1.0, alpha=0.7)
        nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=80,
                               node_color=node_colors, edgecolors="black", linewidths=0.4)

        ax.set_title(f"Shell {n_shells}  ({len(offsets)} neighbors)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(color="#d94a4a", label="Center node"),
        mpatches.Patch(color="#4a90d9", label="Neighbors"),
        mpatches.Patch(color="#e0e0e0", label="Other nodes"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
