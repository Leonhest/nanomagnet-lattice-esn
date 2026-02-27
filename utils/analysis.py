"""Reservoir analysis: multi-panel diagnostic figure + optional eigenvector viz.

Supports both tile JSON files and config.yaml-based reservoirs.
"""

import os
import sys
import copy
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import yaml

# Allow running as ``python -m utils.analysis`` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matrix import Matrix, load_tile_with_metadata, tile_to_lattice, plot_tile
from utils.formula import spectral_radius, calculate_total_recurrent_influence, calculate_avg_degree


# ---------------------------------------------------------------------------
# Generic reservoir analysis
# ---------------------------------------------------------------------------

def analyze_reservoir(W_res, *, G_res=None, tile_G=None, metadata=None,
                      title="Reservoir Analysis", save_path=None, show=True, dpi=300):
    """Produce a multi-panel analysis figure for any reservoir weight matrix.

    Parameters
    ----------
    W_res : np.ndarray
        Square reservoir weight matrix.
    G_res : nx.DiGraph or None
        Full lattice graph (used for degree stats). Built from W_res if None.
    tile_G : nx.DiGraph or None
        Tile-level graph. When provided, tile-specific panels are shown.
    metadata : dict or None
        Arbitrary metadata displayed in the info panel.
    title : str
        Figure suptitle.
    save_path : str or None
        Where to save the PNG.
    show : bool
        Whether to call ``plt.show()``.
    dpi : int
        Figure resolution.

    Returns
    -------
    dict
        Computed statistics for programmatic use.
    """
    has_tile = tile_G is not None

    # --- Build G_res from W_res if not provided ------------------------------
    if G_res is None:
        G_res = nx.DiGraph(W_res)

    # --- Determine weight source for distribution panels ---------------------
    if has_tile:
        weights = np.array([d["weight"] for _, _, d in tile_G.edges(data=True)])
    else:
        weights = W_res[W_res != 0]
        if len(weights) == 0:
            weights = W_res.ravel()

    # --- Precompute statistics -----------------------------------------------
    sr = spectral_radius(W_res)
    tri = calculate_total_recurrent_influence(W_res)
    avg_deg = calculate_avg_degree(W_res)

    # Sign counts
    n_pos = int(np.sum(weights > 0))
    n_neg = int(np.sum(weights < 0))
    n_zero = int(np.sum(weights == 0))

    # Directed edge analysis (skip self-loops)
    source_G = tile_G if has_tile else G_res
    non_self_edges = [(u, v) for u, v, _ in source_G.edges(data=True) if u != v]
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
    self_loops = {u: d["weight"] for u, v, d in source_G.edges(data=True) if u == v}

    # In/out degree and strength from lattice
    in_degrees = np.array([d for _, d in G_res.in_degree()])
    out_degrees = np.array([d for _, d in G_res.out_degree()])
    in_strength = np.sum(np.abs(W_res), axis=0)
    out_strength = np.sum(np.abs(W_res), axis=1)

    # Eigenvalues
    eigvals = np.linalg.eigvals(W_res)

    # --- Figure layout -------------------------------------------------------
    n_rows = 4 if has_tile else 3
    fig = plt.figure(figsize=(18, 22 if has_tile else 17))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.30)

    fig.suptitle(title, fontsize=16, y=0.99)

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

    # Row 1, Col 0: Self-connection weights (tile) or Sparsity heatmap (non-tile)
    ax10 = fig.add_subplot(gs[1, 0])
    if has_tile:
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
    else:
        sparsity = (W_res != 0).astype(float)
        im = ax10.imshow(sparsity, cmap="Greys", aspect="auto", interpolation="nearest")
        ax10.set_title(f"Sparsity Pattern (density={np.mean(sparsity):.3f})")
        ax10.set_xlabel("Column")
        ax10.set_ylabel("Row")
        fig.colorbar(im, ax=ax10, shrink=0.8)

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
    meta_lines = _build_meta_lines(W_res, G_res, sr, avg_deg, tri_finite,
                                   self_loops, metadata, has_tile=has_tile)
    ax22.text(0.05, 0.95, "\n".join(meta_lines), transform=ax22.transAxes,
              fontsize=11, verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax22.set_title("Metadata & Summary")

    # Row 3: Tile graph visualization (full width) — only for tile mode
    if has_tile:
        ax30 = fig.add_subplot(gs[3, :])
        plot_tile(tile_G, title="Tile Graph", ax=ax30)

    # --- Save / show ---------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved analysis figure to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # --- Return stats dict ---------------------------------------------------
    stats = {
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
        "lattice_size": W_res.shape[0],
        "lattice_edges": G_res.number_of_edges(),
        "metadata": metadata,
    }
    return stats


def _build_meta_lines(W_res, G_res, sr, avg_deg, tri_finite, self_loops,
                      metadata, *, has_tile):
    """Build the text lines for the metadata panel."""
    lines = []
    n = W_res.shape[0]
    m = int(np.sqrt(n))
    lines.append(f"Reservoir size: {n} ({m}x{m})")
    lines.append(f"Reservoir edges: {G_res.number_of_edges()}")
    lines.append("")
    lines.append(f"Spectral radius: {sr:.4f}")
    lines.append(f"Avg degree: {avg_deg:.2f}")
    if len(tri_finite) > 0:
        lines.append(f"Avg TRI: {tri_finite.mean():.4f}")
    lines.append(f"Self-connections: {len(self_loops)}")

    if metadata:
        lines.append("")
        for k, v in metadata.items():
            lines.append(f"{k}: {v}")

    return lines


# ---------------------------------------------------------------------------
# Tile wrapper (backwards-compatible)
# ---------------------------------------------------------------------------

def analyze_tile(tile_path, *, lattice_size=400, tile_shape=None,
                 save_path=None, show=True, dpi=300):
    """Produce a multi-panel analysis figure for a saved tile JSON.

    Thin wrapper around :func:`analyze_reservoir`.
    """
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

    if save_path is None:
        stem = os.path.splitext(tile_path)[0]
        save_path = f"{stem}_analysis.png"

    tile_meta = {
        "Tile file": os.path.basename(tile_path),
        "Tile shape": f"{tile_shape[0]}x{tile_shape[1]}",
        "Tile nodes": tile_G.number_of_nodes(),
        "Tile edges": tile_G.number_of_edges(),
    }
    if metadata:
        method = metadata.get("method", "N/A")
        tile_meta["Method"] = method
        nrmse = metadata.get("best_nrmse")
        if nrmse is not None:
            tile_meta["Best NRMSE"] = f"{nrmse:.6f}"
        for k, v in metadata.get("params", {}).items():
            tile_meta[k] = v

    stats = analyze_reservoir(
        W_res,
        G_res=lattice_G,
        tile_G=tile_G,
        metadata=tile_meta,
        title=f"Tile Analysis — {os.path.basename(tile_path)}",
        save_path=save_path,
        show=show,
        dpi=dpi,
    )

    # Augment stats with tile-specific fields
    stats["tile_path"] = tile_path
    stats["tile_shape"] = tile_shape
    stats["tile_nodes"] = tile_G.number_of_nodes()
    stats["tile_edges"] = tile_G.number_of_edges()

    return stats


# ---------------------------------------------------------------------------
# Config-based analysis
# ---------------------------------------------------------------------------

def analyze_from_config(config_path, *, save_dir=None, show=True, eigvec=False,
                        eigvec_only=False):
    """Analyze reservoir(s) defined in a config.yaml.

    Expands grid search lists and produces one analysis per combination.

    Parameters
    ----------
    config_path : str
        Path to a config.yaml file.
    save_dir : str or None
        Directory for output files. Defaults to the config's parent directory.
    show : bool
        Whether to call ``plt.show()``.
    eigvec : bool
        Also generate interactive eigenvector HTML.
    eigvec_only : bool
        Only generate eigenvector HTML, skip the analysis PNG.

    Returns
    -------
    list[dict]
        Statistics dicts, one per grid combination.
    """
    from utils.config_loader import ConfigLoader

    with open(config_path, "r") as f:
        base_config = yaml.safe_load(f)

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(config_path))

    W_args_base = base_config["esn"]["W_args"]
    target_sr = base_config["esn"].get("spectral_radius")

    # Find grid search lists within W_args, using the full prefix so the
    # exclude set in ConfigLoader (e.g. "esn.W_args.W_res_args.tile.shape") matches.
    list_params = ConfigLoader._find_list_parameters(W_args_base, prefix="esn.W_args")
    # Strip the prefix back out so paths are relative to W_args
    list_params = {k.removeprefix("esn.W_args."): v for k, v in list_params.items()}

    if not list_params:
        combos = [{}]
        param_names = []
    else:
        param_names = list(list_params.keys())
        param_values = list(list_params.values())
        combos = [dict(zip(param_names, c)) for c in product(*param_values)]

    all_stats = []
    for combo in combos:
        W_args = copy.deepcopy(W_args_base)
        # Set scalar values for this combination
        for param_path, value in combo.items():
            ConfigLoader._set_nested_value(W_args, param_path, value)

        # Build label from varying params
        if combo:
            label_parts = [f"{k.split('.')[-1]}={v}" for k, v in combo.items()]
            label = "_".join(label_parts)
        else:
            label = "default"

        # Sanitize filename
        safe_label = label.replace("/", "_").replace("\\", "_").replace(" ", "_")

        print(f"\n--- Analyzing: {label} ---")

        # Check if type is a tile path
        wtype = W_args.get("W_res_args", {}).get("type", "")
        stats = {}
        if isinstance(wtype, str) and wtype.endswith(".json"):
            if not eigvec_only:
                save_path = os.path.join(save_dir, f"analysis_{safe_label}.png")
                stats = analyze_tile(
                    wtype,
                    lattice_size=W_args["size"],
                    save_path=save_path,
                    show=show,
                )
            if eigvec or eigvec_only:
                from utils.eigvec_viz import eigenvector_viz_from_tile
                eigvec_path = os.path.join(save_dir, f"eigvec_{safe_label}.html")
                eigenvector_viz_from_tile(wtype, lattice_size=W_args["size"],
                                         target_sr=target_sr,
                                         save_path=eigvec_path)
        else:
            mat = Matrix(W_args)
            W_res_np = mat.W_res.detach().cpu().numpy()
            G_res = getattr(mat, "G_res", None)

            if not eigvec_only:
                meta = {k.split(".")[-1]: v for k, v in combo.items()} if combo else {}
                meta["type"] = wtype

                save_path = os.path.join(save_dir, f"analysis_{safe_label}.png")
                stats = analyze_reservoir(
                    W_res_np,
                    G_res=G_res,
                    metadata=meta,
                    title=f"Reservoir Analysis — {label}",
                    save_path=save_path,
                    show=show,
                )

            if eigvec or eigvec_only:
                from utils.eigvec_viz import eigenvector_viz
                eigvec_path = os.path.join(save_dir, f"eigvec_{safe_label}.html")
                eigenvector_viz(W_res_np, target_sr=target_sr,
                                save_path=eigvec_path,
                                title=f"Eigenvector Explorer — {label}")

        all_stats.append(stats)

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.analysis <tile.json or config.yaml> [--size=N] [--sr=X] [--no-show] [--eigvec] [--eigvec-only]")
        sys.exit(1)

    args = sys.argv[1:]
    input_path = args[0]
    show = "--no-show" not in args
    do_eigvec = "--eigvec" in args
    eigvec_only = "--eigvec-only" in args

    size_arg = next((a for a in args if a.startswith("--size=")), None)
    lattice_size = int(size_arg.split("=")[1]) if size_arg else 400

    sr_arg = next((a for a in args if a.startswith("--sr=")), None)
    target_sr = float(sr_arg.split("=")[1]) if sr_arg else None

    if input_path.endswith(".json"):
        if not eigvec_only:
            stats = analyze_tile(input_path, lattice_size=lattice_size, show=show)
            print(f"\nSpectral radius: {stats['spectral_radius']:.4f}")
            print(f"Avg degree: {stats['avg_degree']:.2f}")
            if stats["avg_tri"] is not None:
                print(f"Avg TRI: {stats['avg_tri']:.4f}")

        if do_eigvec or eigvec_only:
            from utils.eigvec_viz import eigenvector_viz_from_tile
            eigenvector_viz_from_tile(input_path, lattice_size=lattice_size,
                                     target_sr=target_sr)

    elif input_path.endswith(".yaml") or input_path.endswith(".yml"):
        all_stats = analyze_from_config(input_path, show=show, eigvec=do_eigvec,
                                        eigvec_only=eigvec_only)
        print(f"\nGenerated {len(all_stats)} configuration(s).")

    else:
        print(f"Error: unrecognized file type '{input_path}'. Expected .json or .yaml")
        sys.exit(1)
