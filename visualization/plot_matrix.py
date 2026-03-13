"""Visualization of tile graphs and full lattice reservoir graphs.

Usage:
    python -m visualization.plot_matrix                     # random tiled lattice
    python -m visualization.plot_matrix tile.json           # visualize tile only
    python -m visualization.plot_matrix tile.json --lattice # tile onto full lattice
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from matrix import load_tile, load_tile_with_metadata, tile_to_lattice, Matrix


def plot_tile(tile_G, title="Tile graph", save_path=None, show=True, ax=None):
    """Visualize a tile graph with ghost padding nodes for periodic edges.

    Wrapping edges are drawn to ghost nodes in a padding layer around the tile,
    making the periodic structure clear without chaotic cross-tile arcs.

    If *ax* is provided, draws onto that axes instead of creating a new figure.
    In that case save_path/show are ignored (caller manages the figure).
    """
    if tile_G.number_of_edges() == 0:
        print("Tile has no edges to plot.")
        return

    tile_nodes = list(tile_G.nodes())
    rows = sorted(set(n[0] for n in tile_nodes))
    cols = sorted(set(n[1] for n in tile_nodes))
    tile_rows = len(rows)
    tile_cols = len(cols)

    # Build a display graph with ghost nodes for wrapping edges
    display_G = nx.DiGraph()
    for node in tile_nodes:
        display_G.add_node(node, ghost=False)

    ghost_id = 0
    ghost_nodes = set()

    for u, v, d in tile_G.edges(data=True):
        dr_raw = v[0] - u[0]
        dc_raw = v[1] - u[1]
        dr = dr_raw
        dc = dc_raw
        if abs(dr) > tile_rows // 2:
            dr = dr - tile_rows if dr > 0 else dr + tile_rows
        if abs(dc) > tile_cols // 2:
            dc = dc - tile_cols if dc > 0 else dc + tile_cols

        if dr == dr_raw and dc == dc_raw:
            # Normal edge — draw directly
            display_G.add_edge(u, v, weight=d["weight"])
        else:
            # Wrapping edge — draw to a ghost node at the unwrapped position
            ghost_pos = (u[0] + dr, u[1] + dc)
            ghost_key = f"g{ghost_id}"
            ghost_id += 1
            display_G.add_node(ghost_key, ghost=True, display_pos=ghost_pos)
            display_G.add_edge(u, ghost_key, weight=d["weight"])
            ghost_nodes.add(ghost_key)

    # Positions: real nodes at grid coords, ghost nodes at their unwrapped positions
    pos = {}
    for node in display_G.nodes():
        if display_G.nodes[node].get("ghost"):
            gp = display_G.nodes[node]["display_pos"]
            pos[node] = (gp[1], -gp[0])
        else:
            pos[node] = (node[1], -node[0])

    # Separate real and ghost node lists
    real_nodes = [n for n in display_G.nodes() if not display_G.nodes[n].get("ghost")]
    ghost_list = [n for n in display_G.nodes() if display_G.nodes[n].get("ghost")]

    # Edge colors
    edge_weights = [round(d['weight'], 4) for _, _, d in display_G.edges(data=True)]
    w_arr = np.array(edge_weights)
    w_max = max(abs(w_arr.min()), abs(w_arr.max())) or 1.0
    norm = plt.Normalize(vmin=-w_max, vmax=w_max)
    cmap = plt.cm.RdBu_r
    edge_colors = [cmap(norm(w)) for w in edge_weights]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    # Draw real nodes
    nx.draw_networkx_nodes(display_G, pos=pos, nodelist=real_nodes, ax=ax,
                           node_size=400, node_color="lightgray", edgecolors="black")
    nx.draw_networkx_labels(display_G, pos=pos, labels={n: str(n) for n in real_nodes},
                            ax=ax, font_size=9)

    # Draw ghost nodes (smaller, faded)
    if ghost_list:
        nx.draw_networkx_nodes(display_G, pos=pos, nodelist=ghost_list, ax=ax,
                               node_size=200, node_color="white", edgecolors="gray",
                               alpha=0.5, node_shape="s")

    # Draw edges
    nx.draw_networkx_edges(
        display_G, pos=pos, ax=ax, edge_color=edge_colors, width=2,
        connectionstyle="arc3,rad=0.1", arrows=True, arrowsize=12,
    )

    # Edge weight labels
    edge_labels = {(u, v): f"{d['weight']:.3f}" for u, v, d in display_G.edges(data=True)}
    nx.draw_networkx_edge_labels(display_G, pos=pos, edge_labels=edge_labels, ax=ax, font_size=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Edge weight", shrink=0.8)

    ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close()


def plot_lattice(G, title="Tiled lattice reservoir", save_path=None, show=True):
    """Visualize a full lattice reservoir graph with edges colored by unique weight."""
    from matplotlib.lines import Line2D

    pos = {node: (node[1], -node[0]) for node in G.nodes()}

    edge_weights = [round(d['weight'], 6) for _, _, d in G.edges(data=True)]
    unique_weights = sorted(set(edge_weights))
    cmap = plt.cm.get_cmap("tab20", len(unique_weights))
    weight_to_color = {w: cmap(i) for i, w in enumerate(unique_weights)}
    edge_colors = [weight_to_color[w] for w in edge_weights]

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=300)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_colors, width=2,
        connectionstyle="arc3,rad=0.15", arrows=True, arrowsize=15,
    )

    legend_handles = [
        Line2D([0], [0], color=weight_to_color[w], lw=2, label=f"{w:.3f}")
        for w in unique_weights
    ]
    ax.legend(handles=legend_handles, title="Edge weight", loc="upper left", fontsize=7)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # python -m visualization.plot_matrix                     # random tiled lattice
    # python -m visualization.plot_matrix tile.json           # visualize tile only
    # python -m visualization.plot_matrix tile.json --lattice # tile onto full lattice
    # For analysis, use: python -m visualization.analysis <tile.json or config.yaml>
    args = sys.argv[1:]
    json_path = next((a for a in args if a.endswith(".json")), None)
    show_lattice = "--lattice" in args

    if json_path and show_lattice:
        tile_G, meta = load_tile_with_metadata(json_path)
        tile_shape = meta.get("tile_shape", [3, 3])
        lattice_G = tile_to_lattice(tile_G, tile_shape, lattice_size=36)
        plot_lattice(lattice_G, title=f"Lattice from {json_path}")
    elif json_path:
        tile_G = load_tile(json_path)
        plot_tile(tile_G, title=f"Tile: {json_path}")
    else:
        matrix = Matrix({
            "size": 36,
            "W_in_args": {"input_scale": 1, "distribution": "uniform"},
            "W_res_args": {
                "self_connection": 0.0,
                "sign_frac": 0.5,
                "type": "tile",
                "neighborhood": "Moore",
                "directed_edges_weights": 0.0,
                "directed_edges_fraction": 0.5,
                "tile": {"shape": [3, 3], "method": "random"},
            },
        })
        plot_lattice(matrix.G_res)
