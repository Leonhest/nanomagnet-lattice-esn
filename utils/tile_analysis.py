"""Tile analysis: load a tile JSON and produce a multi-panel diagnostic figure."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Allow running as ``python -m utils.tile_analysis`` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matrix import load_tile_with_metadata, tile_to_lattice, plot_tile
from utils.formula import spectral_radius, calculate_total_recurrent_influence, calculate_avg_degree


def analyze_tile(tile_path, *, lattice_size=400, tile_shape=None, save_path=None, show=True, dpi=300):
    """Produce a multi-panel analysis figure for a saved tile JSON.

    Parameters
    ----------
    tile_path : str
        Path to a tile JSON file.
    lattice_size : int
        Number of nodes in the tiled lattice (must be a perfect square).
    tile_shape : tuple[int, int] or None
        Tile dimensions (rows, cols). Auto-detected from metadata, falls back to (3, 3).
    save_path : str or None
        Where to save the PNG. Defaults to ``<tile_stem>_analysis.png`` next to the JSON.
    show : bool
        Whether to call ``plt.show()``.
    dpi : int
        Figure resolution.

    Returns
    -------
    dict
        Computed statistics for programmatic use.
    """
    # --- Load tile and build lattice -----------------------------------------
    tile_G, metadata = load_tile_with_metadata(tile_path)

    if tile_shape is None:
        ts = metadata.get("tile_shape")
        if ts:
            tile_shape = tuple(ts)
        else:
            nodes = list(tile_G.nodes())
            rows = max(n[0] for n in nodes) + 1
            cols = max(n[1] for n in nodes) + 1
            tile_shape = (rows, cols)

    lattice_G = tile_to_lattice(tile_G, tile_shape, lattice_size=lattice_size)
    W_res = nx.to_numpy_array(lattice_G)

    # --- Precompute statistics -----------------------------------------------
    weights = np.array([d["weight"] for _, _, d in tile_G.edges(data=True)])
    sr = spectral_radius(W_res)
    tri = calculate_total_recurrent_influence(W_res)
    avg_deg = calculate_avg_degree(W_res)

    # Sign counts
    n_pos = int(np.sum(weights > 0))
    n_neg = int(np.sum(weights < 0))
    n_zero = int(np.sum(weights == 0))

    # Directed edge analysis (skip self-loops)
    non_self_edges = [(u, v) for u, v, _ in tile_G.edges(data=True) if u != v]
    edge_set = set(non_self_edges)
    bidirectional = 0
    unidirectional = 0
    counted = set()
    for u, v in non_self_edges:
        pair = frozenset((u, v))
        if pair in counted:
            continue
        counted.add(pair)
        if (v, u) in edge_set:
            bidirectional += 1
        else:
            unidirectional += 1

    # Self-connections
    self_loops = {u: d["weight"] for u, v, d in tile_G.edges(data=True) if u == v}

    # In/out degree and strength from lattice
    in_degrees = np.array([d for _, d in lattice_G.in_degree()])
    out_degrees = np.array([d for _, d in lattice_G.out_degree()])
    in_strength = np.sum(np.abs(W_res), axis=0)   # sum abs of incoming weights per node
    out_strength = np.sum(np.abs(W_res), axis=1)   # sum abs of outgoing weights per node

    # Eigenvalues for spectrum plot
    eigvals = np.linalg.eigvals(W_res)

    # --- Create figure -------------------------------------------------------
    fig = plt.figure(figsize=(18, 22))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.30)

    # Row 0, Col 0: Weight distribution histogram
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.hist(weights, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    ax00.axvline(weights.mean(), color="red", linestyle="--", label=f"mean={weights.mean():.3f}")
    ax00.set_title("Weight Distribution")
    ax00.set_xlabel("Weight")
    ax00.set_ylabel("Count")
    ax00.legend(fontsize=8)
    stats_text = f"std={weights.std():.3f}\nzeros={n_zero}"
    ax00.text(0.97, 0.95, stats_text, transform=ax00.transAxes, fontsize=8,
              verticalalignment="top", horizontalalignment="right",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Row 0, Col 1: Sign distribution bar chart
    ax01 = fig.add_subplot(gs[0, 1])
    bars = ax01.bar(["+", "-", "0"], [n_pos, n_neg, n_zero],
                    color=["dodgerblue", "tomato", "gray"], edgecolor="black")
    for bar, val in zip(bars, [n_pos, n_neg, n_zero]):
        ax01.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                  str(val), ha="center", fontsize=10)
    ax01.set_title("Sign Distribution")
    ax01.set_ylabel("Count")

    # Row 0, Col 2: Directed edge distribution
    ax02 = fig.add_subplot(gs[0, 2])
    bars = ax02.bar(["Bidirectional", "Unidirectional"], [bidirectional, unidirectional],
                    color=["mediumpurple", "sandybrown"], edgecolor="black")
    for bar, val in zip(bars, [bidirectional, unidirectional]):
        ax02.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                  str(val), ha="center", fontsize=10)
    ax02.set_title("Directed Edge Pairs (excl. self-loops)")
    ax02.set_ylabel("Count")

    # Row 1, Col 0: Self-connection weights
    ax10 = fig.add_subplot(gs[1, 0])
    tile_nodes = sorted(tile_G.nodes())
    if self_loops:
        labels = [str(n) for n in tile_nodes if n in self_loops]
        vals = [self_loops[n] for n in tile_nodes if n in self_loops]
        colors = ["dodgerblue" if v >= 0 else "tomato" for v in vals]
        ax10.bar(labels, vals, color=colors, edgecolor="black")
        ax10.set_ylabel("Weight")
        ax10.axhline(0, color="black", linewidth=0.5)
    else:
        ax10.text(0.5, 0.5, "No self-connections", transform=ax10.transAxes,
                  ha="center", va="center", fontsize=14, color="gray")
    ax10.set_title("Self-Connection Weights")

    # Row 1, Col 1: Eigenvalue spectrum
    ax11 = fig.add_subplot(gs[1, 1])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax11.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")
    ax11.plot(sr * np.cos(theta), sr * np.sin(theta), "r-", alpha=0.5,
              label=f"SR={sr:.4f}")
    ax11.scatter(eigvals.real, eigvals.imag, s=6, alpha=0.5, color="steelblue")
    ax11.set_aspect("equal")
    ax11.set_title("Eigenvalue Spectrum")
    ax11.set_xlabel("Re")
    ax11.set_ylabel("Im")
    ax11.legend(fontsize=8)
    ax11.grid(True, alpha=0.3)

    # Row 1, Col 2: TRI distribution
    ax12 = fig.add_subplot(gs[1, 2])
    tri_finite = tri[np.isfinite(tri)]
    if len(tri_finite) > 0:
        ax12.hist(tri_finite, bins=30, color="seagreen", edgecolor="black", alpha=0.8)
        ax12.axvline(tri_finite.mean(), color="red", linestyle="--",
                     label=f"mean={tri_finite.mean():.4f}")
        ax12.legend(fontsize=8)
    ax12.set_title("TRI Distribution")
    ax12.set_xlabel("Total Recurrent Influence")
    ax12.set_ylabel("Count")

    # Row 2, Col 0: In/out degree histogram
    ax20 = fig.add_subplot(gs[2, 0])
    bins = np.arange(min(in_degrees.min(), out_degrees.min()) - 0.5,
                     max(in_degrees.max(), out_degrees.max()) + 1.5, 1)
    ax20.hist(in_degrees, bins=bins, alpha=0.6, color="steelblue", edgecolor="black", label="In-degree")
    ax20.hist(out_degrees, bins=bins, alpha=0.6, color="orange", edgecolor="black", label="Out-degree")
    ax20.set_title("In/Out Degree Distribution")
    ax20.set_xlabel("Degree")
    ax20.set_ylabel("Count")
    ax20.legend(fontsize=8)

    # Row 2, Col 1: In/out strength histogram
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.hist(in_strength, bins=30, alpha=0.6, color="steelblue", edgecolor="black", label="In-strength")
    ax21.hist(out_strength, bins=30, alpha=0.6, color="orange", edgecolor="black", label="Out-strength")
    ax21.set_title("In/Out Strength Distribution")
    ax21.set_xlabel("Sum of |weights|")
    ax21.set_ylabel("Count")
    ax21.legend(fontsize=8)

    # Row 2, Col 2: Metadata text panel
    ax22 = fig.add_subplot(gs[2, 2])
    ax22.axis("off")
    meta_lines = []
    meta_lines.append(f"Tile file: {os.path.basename(tile_path)}")
    meta_lines.append(f"Tile shape: {tile_shape[0]}x{tile_shape[1]}")
    meta_lines.append(f"Tile nodes: {tile_G.number_of_nodes()}")
    meta_lines.append(f"Tile edges: {tile_G.number_of_edges()}")
    meta_lines.append(f"Lattice size: {lattice_size} ({int(np.sqrt(lattice_size))}x{int(np.sqrt(lattice_size))})")
    meta_lines.append(f"Lattice edges: {lattice_G.number_of_edges()}")
    meta_lines.append("")
    meta_lines.append(f"Spectral radius: {sr:.4f}")
    meta_lines.append(f"Avg degree: {avg_deg:.2f}")
    if len(tri_finite) > 0:
        meta_lines.append(f"Avg TRI: {tri_finite.mean():.4f}")
    meta_lines.append(f"Self-connections: {len(self_loops)}")
    if metadata:
        meta_lines.append("")
        method = metadata.get("method", "N/A")
        meta_lines.append(f"Method: {method}")
        nrmse = metadata.get("best_nrmse")
        if nrmse is not None:
            meta_lines.append(f"Best NRMSE: {nrmse:.6f}")
        params = metadata.get("params", {})
        for k, v in params.items():
            meta_lines.append(f"  {k}: {v}")

    ax22.text(0.05, 0.95, "\n".join(meta_lines), transform=ax22.transAxes,
              fontsize=11, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax22.set_title("Metadata & Summary")

    # Row 3: Tile graph visualization (spanning all 3 columns)
    ax30 = fig.add_subplot(gs[3, :])
    plot_tile(tile_G, title="Tile Graph", ax=ax30)

    # --- Save / show ---------------------------------------------------------
    if save_path is None:
        stem = os.path.splitext(tile_path)[0]
        save_path = f"{stem}_analysis.png"

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved analysis figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Return stats dict ---------------------------------------------------
    stats = {
        "tile_path": tile_path,
        "tile_shape": tile_shape,
        "tile_nodes": tile_G.number_of_nodes(),
        "tile_edges": tile_G.number_of_edges(),
        "lattice_size": lattice_size,
        "lattice_edges": lattice_G.number_of_edges(),
        "spectral_radius": sr,
        "avg_degree": avg_deg,
        "avg_tri": float(tri_finite.mean()) if len(tri_finite) > 0 else None,
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_zero": n_zero,
        "n_self_loops": len(self_loops),
        "bidirectional_pairs": bidirectional,
        "unidirectional_pairs": unidirectional,
        "metadata": metadata,
    }
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.tile_analysis <tile.json> [--size=N] [--no-show]")
        sys.exit(1)

    args = sys.argv[1:]
    json_path = next((a for a in args if a.endswith(".json")), None)
    if json_path is None:
        print("Error: no .json tile path provided")
        sys.exit(1)

    size_arg = next((a for a in args if a.startswith("--size=")), None)
    lattice_size = int(size_arg.split("=")[1]) if size_arg else 400
    show = "--no-show" not in args

    stats = analyze_tile(json_path, lattice_size=lattice_size, show=show)
    print(f"\nSpectral radius: {stats['spectral_radius']:.4f}")
    print(f"Avg degree: {stats['avg_degree']:.2f}")
    if stats["avg_tri"] is not None:
        print(f"Avg TRI: {stats['avg_tri']:.4f}")
