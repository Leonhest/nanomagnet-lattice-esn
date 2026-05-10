"""Generate background/methods figures for the thesis.

Usage:
    python -m visualization.generate_thesis_figures --output_dir ../thesis/Figures/methods/
    python -m visualization.generate_thesis_figures --figure topology
    python -m visualization.generate_thesis_figures --figure fold
    python -m visualization.generate_thesis_figures --figure chaos
    python -m visualization.generate_thesis_figures --figure tri
    python -m visualization.generate_thesis_figures --figure all
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from math import sqrt

# ---------- Shared style ----------

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
}

# Consistent palette
C_BLUE = "#2166AC"
C_RED = "#B2182B"
C_ORANGE = "#E08214"
C_GREEN = "#1B7837"
C_GREY = "#999999"
C_LIGHT_GREY = "#DDDDDD"
C_NODE = "#4393C3"
C_NODE_EDGE = "#333333"


def apply_style():
    plt.rcParams.update(STYLE)


# ---------- 1. Topology comparison ----------

def fig_topology_comparison(output_dir, dpi=300):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    N = 36
    seed = 42
    node_kw = dict(node_size=35, node_color=C_NODE, edgecolors=C_NODE_EDGE,
                   linewidths=0.4)
    edge_kw = dict(edge_color="#AAAAAA", width=0.4, alpha=0.5)

    # Random sparse
    ax = axes[0]
    G = nx.erdos_renyi_graph(N, 0.12, seed=seed, directed=True)
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, **edge_kw)
    ax.set_title("Random sparse")
    ax.axis("off")

    # Ring / Cycle
    ax = axes[1]
    G = nx.cycle_graph(N, create_using=nx.DiGraph)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=5,
                           arrowstyle="-|>", **edge_kw)
    ax.set_title("Ring / cycle")
    ax.axis("off")

    # 2D Lattice (Moore)
    ax = axes[2]
    side = int(sqrt(N))
    G = nx.grid_2d_graph(side, side)
    for r in range(side):
        for c in range(side):
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    G.add_edge((r, c), (nr, nc))
    pos = {(r, c): (c, -r) for r in range(side) for c in range(side)}
    nx.draw_networkx_nodes(G, pos, ax=ax, **node_kw)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=False, **edge_kw)
    ax.set_title("2D lattice (Moore)")
    ax.axis("off")

    fig.tight_layout(pad=0.5)
    path = os.path.join(output_dir, "topology_comparison.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------- 2. Fold bifurcation / S-curve (for 2.6) ----------

def fig_fold_bifurcation(output_dir, dpi=300):
    apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Parametric S-curve: fold normal form
    t = np.linspace(-1.4, 1.4, 600)
    mu = t**3 - t + 0.9
    x = t * 0.5 + 0.5

    dmu = np.gradient(mu, t)
    fold_idx = np.where(np.diff(np.sign(dmu)))[0]

    if len(fold_idx) >= 2:
        i1, i2 = fold_idx[0], fold_idx[1]

        # Branches
        ax.plot(mu[:i1+1], x[:i1+1], color=C_BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(mu[i1:i2+1], x[i1:i2+1], color=C_RED, linewidth=1.8,
                linestyle="--", dash_capstyle="round")
        ax.plot(mu[i2:], x[i2:], color=C_BLUE, linewidth=2, solid_capstyle="round")

        # Fold points
        ax.plot(mu[i1], x[i1], 'o', color=C_RED, markersize=6, zorder=5)
        ax.plot(mu[i2], x[i2], 'o', color=C_RED, markersize=6, zorder=5)

        # i1 = lower-right fold (x~0.2, mu~1.3): jump UP to upper branch
        # i2 = upper-left fold (x~0.8, mu~0.5): jump DOWN to lower branch
        ax.annotate("", xy=(mu[i1], x[i1] + 0.55),
                    xytext=(mu[i1], x[i1] + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, linewidth=1.8))
        ax.annotate("", xy=(mu[i2], x[i2] - 0.55),
                    xytext=(mu[i2], x[i2] - 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, linewidth=1.8))

        # Bistable shading
        ax.axvspan(mu[i2], mu[i1], alpha=0.06, color=C_RED, zorder=0)

        # Labels
        ax.text((mu[i1] + mu[i2]) / 2, 0.03, "Bistable\nregion",
                ha="center", fontsize=8, color=C_RED, fontstyle="italic")
        ax.text(mu[i1] + 0.04, x[i1] + 0.28, "jump",
                fontsize=8, color=C_ORANGE, fontstyle="italic")
        ax.text(mu[i2] + 0.04, x[i2] - 0.35, "jump",
                fontsize=8, color=C_ORANGE, fontstyle="italic")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C_BLUE, linewidth=2, label="Stable"),
        Line2D([0], [0], color=C_RED, linewidth=1.8, linestyle="--", label="Unstable"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False)

    ax.set_xlabel("Control parameter")
    ax.set_ylabel("System state")
    ax.set_xlim(0.25, 1.55)

    fig.tight_layout()
    path = os.path.join(output_dir, "fold_bifurcation.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------- 3. Period-doubling / route to chaos (for 2.2) ----------

def fig_period_doubling(output_dir, dpi=300):
    """Logistic map bifurcation diagram showing route to chaos."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))

    r_min, r_max = 2.8, 4.0
    r_values = np.linspace(r_min, r_max, 2000)
    n_settle = 300
    n_plot = 100

    for r in r_values:
        x = 0.5
        for _ in range(n_settle):
            x = r * x * (1 - x)
        xs = []
        for _ in range(n_plot):
            x = r * x * (1 - x)
            xs.append(x)
        ax.plot([r] * n_plot, xs, ',', color=C_BLUE, markersize=0.2, alpha=0.4)

    # Annotations
    ax.axvline(x=3.0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.5)
    ax.axvline(x=3.57, color=C_RED, linewidth=0.5, linestyle=":", alpha=0.5)
    ax.text(3.0, 1.02, "Period\ndoubling", ha="center", fontsize=7,
            color=C_GREY, fontstyle="italic", transform=ax.get_xaxis_transform())
    ax.text(3.57, 1.02, "Onset of\nchaos", ha="center", fontsize=7,
            color=C_RED, fontstyle="italic", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Control parameter $r$")
    ax.set_ylabel("Attractor values $x^*$")
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    path = os.path.join(output_dir, "period_doubling_bifurcation.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------- 4. TRI ratio schematic ----------

def fig_tri_schematic(output_dir, dpi=300):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8),
                              gridspec_kw={"width_ratios": [1.2, 1]})
    side = 5

    # --- Left: Self-return paths ---
    ax = axes[0]
    center = (2, 2)

    G = nx.grid_2d_graph(side, side)
    pos = {(r, c): (c, -r) for r in range(side) for c in range(side)}

    # Color by distance
    colors = []
    sizes = []
    for node in G.nodes():
        d = abs(node[0] - center[0]) + abs(node[1] - center[1])
        if node == center:
            colors.append(C_RED)
            sizes.append(100)
        elif d <= 1:
            colors.append(C_ORANGE)
            sizes.append(60)
        elif d <= 2:
            colors.append("#FDD49E")
            sizes.append(45)
        else:
            colors.append(C_LIGHT_GREY)
            sizes.append(35)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors=C_NODE_EDGE, linewidths=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CCCCCC", width=0.4)

    # Return paths
    paths = [
        ([(2,2), (2,1), (2,2)], C_RED, 2.0),
        ([(2,2), (1,2), (2,2)], C_RED, 2.0),
        ([(2,2), (2,3), (1,3), (1,2), (2,2)], "#C994C7", 1.8),
    ]
    for path_nodes, pc, lw in paths:
        edges = list(zip(path_nodes[:-1], path_nodes[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                               edge_color=pc, width=lw, alpha=0.7,
                               arrows=True, arrowsize=8,
                               arrowstyle="-|>", connectionstyle="arc3,rad=0.15")

    ax.set_title("Self-return paths $\\rightarrow R_{jj}$", fontsize=11)
    ax.axis("off")

    legend_elements = [
        mpatches.Patch(color=C_RED, label="Target node $j$"),
        mpatches.Patch(color=C_ORANGE, label="1-hop neighbors"),
        mpatches.Patch(color="#FDD49E", label="2-hop neighbors"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8,
              frameon=False)

    # --- Right: TRI ratio bar comparison ---
    ax = axes[1]
    categories = ["Lattice\n(local)", "Random\n(global)"]
    self_inf = [0.27, 0.05]
    other_inf = [0.73, 0.95]
    x_pos = np.arange(len(categories))
    width = 0.45

    ax.bar(x_pos, self_inf, width, color=C_RED, label="Self-influence $R_{jj}$",
           zorder=3, edgecolor="white", linewidth=0.5)
    ax.bar(x_pos, other_inf, width, bottom=self_inf, color=C_LIGHT_GREY,
           label="Cross-influence", zorder=3, edgecolor="white", linewidth=0.5)

    for i, s in enumerate(self_inf):
        ax.text(i, s / 2, f"{s:.0%}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")

    ax.set_ylabel("Fraction of total influence")
    ax.set_title("TRI ratio comparison", fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout(pad=1.5)
    path = os.path.join(output_dir, "tri_ratio_schematic.pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Generate background/methods figures for the thesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str,
                        default="../thesis/Figures/methods/",
                        help="Directory to save figures")
    parser.add_argument("--figure", type=str, default="all",
                        choices=["topology", "fold", "chaos", "tri", "all"],
                        help="Which figure to generate")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.figure in ("topology", "all"):
        fig_topology_comparison(args.output_dir, args.dpi)
    if args.figure in ("fold", "all"):
        fig_fold_bifurcation(args.output_dir, args.dpi)
    if args.figure in ("chaos", "all"):
        fig_period_doubling(args.output_dir, args.dpi)
    if args.figure in ("tri", "all"):
        fig_tri_schematic(args.output_dir, args.dpi)

    print("Done!")


if __name__ == "__main__":
    main()
