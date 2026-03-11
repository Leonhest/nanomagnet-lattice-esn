"""
Thesis-quality 3-panel figure illustrating tile tiling onto a lattice.

(a) Single 3×3 tile with directed weighted edges
(b) Tile placed on top-left corner of 12×12 lattice
(c) Full tiled 12×12 lattice

Usage:
    python -m utils.plot_tile_tiling
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import networkx as nx

from matrix import load_tile_with_metadata, tile_to_lattice

TILE_PATH = "experiments/conference/methods/best_tile_cmaes.json"
OUTPUT_PATH = "experiments/conference/methods/tile_tiling_illustration.png"
LATTICE_SIZE = 144  # 12×12
LATTICE_DIM = 12
TILE_ROWS, TILE_COLS = 3, 3
TILE_NODE_PALETTE = plt.cm.tab10(np.linspace(0, 1, TILE_ROWS * TILE_COLS))


def _tile_color(node):
    """Return the color for a node based on its tile-coordinate (modulo)."""
    return TILE_NODE_PALETTE[(node[0] % TILE_ROWS) * TILE_COLS + (node[1] % TILE_COLS)]


def _get_shared_norm(tile_G, lattice_G):
    """Compute a shared symmetric color normalization from all edge weights."""
    all_weights = [d["weight"] for *_, d in tile_G.edges(data=True)]
    all_weights += [d["weight"] for *_, d in lattice_G.edges(data=True)]
    w_max = max(abs(min(all_weights)), abs(max(all_weights))) or 1.0
    return Normalize(vmin=-w_max, vmax=w_max)


def _grid_pos(node):
    """Map (row, col) node to (x, y) plot coordinates."""
    return (node[1], -node[0])


def _draw_edges(G, pos, ax, norm, cmap, width=1.5, arrowsize=10, rad=0.1, alpha=1.0):
    """Draw directed edges colored by weight."""
    if G.number_of_edges() == 0:
        return
    edge_colors = [cmap(norm(d["weight"])) for _, _, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_colors, width=width,
        connectionstyle=f"arc3,rad={rad}", arrows=True, arrowsize=arrowsize,
        alpha=alpha,
    )


def panel_a_tile(ax, tile_G, norm, cmap):
    """Panel (a): Single tile with ghost nodes for wrapping edges."""
    tile_nodes = list(tile_G.nodes())
    tile_rows = max(n[0] for n in tile_nodes) + 1
    tile_cols = max(n[1] for n in tile_nodes) + 1

    # Build display graph with ghost nodes for wrapping edges
    display_G = nx.DiGraph()
    for node in tile_nodes:
        display_G.add_node(node, ghost=False)

    ghost_id = 0
    for u, v, d in tile_G.edges(data=True):
        dr_raw = v[0] - u[0]
        dc_raw = v[1] - u[1]
        dr, dc = dr_raw, dc_raw
        if abs(dr) > tile_rows // 2:
            dr = dr - tile_rows if dr > 0 else dr + tile_rows
        if abs(dc) > tile_cols // 2:
            dc = dc - tile_cols if dc > 0 else dc + tile_cols

        if dr == dr_raw and dc == dc_raw:
            display_G.add_edge(u, v, weight=d["weight"])
        else:
            ghost_pos = (u[0] + dr, u[1] + dc)
            ghost_key = f"g{ghost_id}"
            ghost_id += 1
            display_G.add_node(ghost_key, ghost=True, display_pos=ghost_pos)
            display_G.add_edge(u, ghost_key, weight=d["weight"])

    # Positions
    pos = {}
    for node in display_G.nodes():
        if display_G.nodes[node].get("ghost"):
            gp = display_G.nodes[node]["display_pos"]
            pos[node] = (gp[1], -gp[0])
        else:
            pos[node] = _grid_pos(node)

    real_nodes = [n for n in display_G if not display_G.nodes[n].get("ghost")]
    ghost_nodes = [n for n in display_G if display_G.nodes[n].get("ghost")]

    # Draw real nodes with tile-coordinate colors
    real_colors = [_tile_color(n) for n in real_nodes]
    nx.draw_networkx_nodes(display_G, pos=pos, nodelist=real_nodes, ax=ax,
                           node_size=350, node_color=real_colors, edgecolors="black", linewidths=1.2)

    # Draw ghost nodes
    if ghost_nodes:
        nx.draw_networkx_nodes(display_G, pos=pos, nodelist=ghost_nodes, ax=ax,
                               node_size=150, node_color="white", edgecolors="gray",
                               alpha=0.5, node_shape="s")

    # Draw edges
    _draw_edges(display_G, pos, ax, norm, cmap, width=1.8, arrowsize=10, rad=0.12)

    ax.set_title("(a) Tile", fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_b_placed(ax, tile_G, norm, cmap):
    """Panel (b): Full 12×12 grid with only top-left 3×3 region active."""
    m = LATTICE_DIM

    # Draw all nodes
    all_nodes = [(r, c) for r in range(m) for c in range(m)]
    pos = {n: _grid_pos(n) for n in all_nodes}

    tile_nodes = list(tile_G.nodes())
    tile_rows = max(n[0] for n in tile_nodes) + 1
    tile_cols = max(n[1] for n in tile_nodes) + 1

    highlight = {(r, c) for r in range(tile_rows) for c in range(tile_cols)}
    inactive = [n for n in all_nodes if n not in highlight]
    active = [n for n in all_nodes if n in highlight]

    # Inactive nodes: light gray
    nx.draw_networkx_nodes(nx.DiGraph(), pos=pos, nodelist=inactive, ax=ax,
                           node_size=30, node_color="#e0e0e0", edgecolors="#cccccc", linewidths=0.5)

    # Active nodes: colored by tile coordinate
    active_colors = [_tile_color(n) for n in active]
    nx.draw_networkx_nodes(nx.DiGraph(), pos=pos, nodelist=active, ax=ax,
                           node_size=80, node_color=active_colors, edgecolors="black", linewidths=1.0)

    # Build the subgraph for the top-left tile region using _tile_edges_to_lattice logic
    # but only for edges within the tile region (no wrapping on the lattice)
    sub_G = nx.DiGraph()
    for n in active:
        sub_G.add_node(n)
    for u, v, d in tile_G.edges(data=True):
        dr_raw = v[0] - u[0]
        dc_raw = v[1] - u[1]
        dr, dc = dr_raw, dc_raw
        if abs(dr) > tile_rows // 2:
            dr = dr - tile_rows if dr > 0 else dr + tile_rows
        if abs(dc) > tile_cols // 2:
            dc = dc - tile_cols if dc > 0 else dc + tile_cols
        tr = u[0] + dr
        tc = u[1] + dc
        if 0 <= tr < tile_rows and 0 <= tc < tile_cols:
            sub_G.add_edge(u, (tr, tc), weight=d["weight"])

    _draw_edges(sub_G, pos, ax, norm, cmap, width=1.0, arrowsize=6, rad=0.15)

    # Dashed rectangle around tile region
    rect = mpatches.FancyBboxPatch(
        (-0.4, -(tile_rows - 1) - 0.4),  # (x, y) in plot coords
        tile_cols - 1 + 0.8, tile_rows - 1 + 0.8,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor="#333333", facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)

    ax.set_title("(b) Tile placed on lattice", fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def panel_c_full(ax, lattice_G, norm, cmap):
    """Panel (c): Full tiled 12×12 lattice with modulo node coloring."""
    pos = {n: _grid_pos(n) for n in lattice_G.nodes()}

    node_list = list(lattice_G.nodes())
    node_colors = [_tile_color(n) for n in node_list]

    nx.draw_networkx_nodes(lattice_G, pos=pos, nodelist=node_list, ax=ax,
                           node_size=40, node_color=node_colors, edgecolors="gray", linewidths=0.4)

    _draw_edges(lattice_G, pos, ax, norm, cmap, width=0.4, arrowsize=3, rad=0.15, alpha=0.7)

    ax.set_title("(c) Tiled lattice", fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    tile_G, meta = load_tile_with_metadata(TILE_PATH)
    tile_shape = meta["tile_shape"]
    lattice_G = tile_to_lattice(tile_G, tile_shape, lattice_size=LATTICE_SIZE)

    cmap = plt.cm.RdBu_r
    norm = _get_shared_norm(tile_G, lattice_G)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))

    panel_a_tile(axes[0], tile_G, norm, cmap)
    panel_b_placed(axes[1], tile_G, norm, cmap)
    panel_c_full(axes[2], lattice_G, norm, cmap)

    # Shared colorbar on the right
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), location="right", shrink=0.7, pad=0.02)
    cbar.set_label("Edge weight", fontsize=12)

    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
