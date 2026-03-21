"""
Figure illustrating a lattice reservoir with self-connections.

4×4 Von Neumann neighborhood lattice with self-loops highlighted.

Usage:
    python -m visualization.plot_lattice_hysteresis
"""

import matplotlib.pyplot as plt
import networkx as nx

OUTPUT_PATH = "experiments/conference/methods/lattice_hysteresis_illustration.png"
LATTICE_DIM = 4


def panel_a_lattice(ax):
    """4×4 Von Neumann lattice with self-loops visualized."""
    m = LATTICE_DIM

    # Build Von Neumann neighborhood graph (4-connectivity)
    G = nx.grid_graph([m, m])
    G = G.to_directed()

    # Add self-connections with a distinct weight
    for node in list(G.nodes()):
        G.add_edge(node, node, weight=1.0)
    # Set neighbor weights
    for u, v, d in G.edges(data=True):
        if u != v:
            d["weight"] = 0.5

    pos = {nd: (nd[1], -nd[0]) for nd in G.nodes()}

    # Color edges by weight — self-loops (1.0) vs neighbors (0.5)
    edge_colors = ["#d94a4a" if u == v else "#888888" for u, v in G.edges()]
    edge_widths = [2.0 if u == v else 0.8 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=350,
                           node_color="#4a90d9", edgecolors="black", linewidths=1.2)
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_colors, width=edge_widths,
        connectionstyle="arc3,rad=0.15", arrows=True, arrowsize=10,
    )

    ax.set_title("Lattice with self-connections", fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a90d9',
               markeredgecolor='black', markersize=10, label='Node'),
        Line2D([0], [0], color='#cccccc', lw=1.5, label='Neighbor connections'),
        Line2D([0], [0], color='#d94a4a', lw=1.5, label='Self-connection'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=8,
              framealpha=0.9, ncol=3, bbox_to_anchor=(0.5, -0.05))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    panel_a_lattice(ax)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
