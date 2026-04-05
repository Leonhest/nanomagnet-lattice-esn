"""Hysteresis sweep comparing CMA-ES tiles vs constant vs baseline-esn.

Each CMA-ES tile is matched to the spectral radius it was trained at.
Produces two figures: node states and RMS/gap.

Usage:
    python -m visualization.plot_tile_hysteresis_sweep experiments/v2/tiles/hysteresis/
    python -m visualization.plot_tile_hysteresis_sweep experiments/v2/tiles/hysteresis/ --save sweep.png
    python -m visualization.plot_tile_hysteresis_sweep experiments/v2/tiles/hysteresis/ --rho 0.95 1.0 1.2
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from activation import Hysteresis
from matrix import Matrix, load_tile_with_metadata


def _parse_spectral_radius_from_filename(path):
    """Extract spectral_radius value from a tile filename like best_tile_cmaes_spectral_radius=1.2.json."""
    basename = os.path.basename(path)
    m = re.search(r'spectral_radius=([0-9]+(?:\.[0-9]+)?)', basename)
    if m:
        return float(m.group(1))
    return None


def _find_tile_files(tile_dir):
    """Find tile JSONs and map spectral_radius -> path."""
    tiles = {}
    for path in sorted(glob.glob(os.path.join(tile_dir, "*.json"))):
        rho = _parse_spectral_radius_from_filename(path)
        if rho is not None:
            tiles[rho] = path
    return tiles


def build_reservoir_from_type(res_type, size, neighborhood, self_conn,
                              sign_frac, dir_frac, dir_weights, rho):
    """Build a reservoir matrix for constant/baseline-esn types."""
    W_args = {
        "size": size,
        "W_in_args": {"input_scale": 0.1, "distribution": "fixed"},
        "W_res_args": {
            "type": res_type,
            "neighborhood": neighborhood,
            "self_connection": self_conn,
            "sign_frac": sign_frac,
            "directed_edges_fraction": dir_frac,
            "directed_edges_weights": dir_weights,
            "tile": {"shape": [3, 3], "method": "random"},
            "alternating_proj": 0,
            "orthogonal_tile": False,
        },
    }
    mat = Matrix(W_args)

    if rho is not None:
        spec_rad = float(torch.max(torch.abs(torch.linalg.eigvals(mat.W_res))).real)
        if spec_rad > 0:
            mat.W_res *= rho / spec_rad

    return mat


def build_reservoir_from_tile(tile_path, size, neighborhood, self_conn, rho):
    """Build a reservoir matrix from a saved tile JSON."""
    tile_data, tile_meta = load_tile_with_metadata(tile_path)

    W_args = {
        "size": size,
        "W_in_args": {"input_scale": 0.1, "distribution": "fixed"},
        "W_res_args": {
            "type": tile_path,
            "neighborhood": neighborhood,
            "self_connection": self_conn,
            "sign_frac": 0.5,
            "directed_edges_fraction": 0.9,
            "directed_edges_weights": 0.0,
            "alternating_proj": 0,
            "orthogonal_tile": False,
        },
    }
    skip_self = tile_meta.get("optimize_self_connections", False)
    mat = Matrix.from_tile(tile_data, W_args, skip_self_connection=skip_self)

    if rho is not None:
        spec_rad = float(torch.max(torch.abs(torch.linalg.eigvals(mat.W_res))).real)
        if spec_rad > 0:
            mat.W_res *= rho / spec_rad

    return mat


def sweep(mat, f, input_values, settle_steps):
    """Run the reservoir to steady state at each constant input value."""
    size = mat.W_res.shape[0]
    x = torch.zeros(size)
    f.reset(size)
    states = np.zeros((len(input_values), size))

    for i, u_val in enumerate(input_values):
        u = torch.tensor(u_val, dtype=torch.float32)
        for _ in range(settle_steps):
            state_input = mat.W_in * u + mat.W_res.mv(x)
            x = f(state_input)
        states[i] = x.detach().numpy()

    return states


def plot_nodes(ax, input_values, states_up, states_down, title, num_nodes=20):
    """Plot individual node activations for sweep up/down."""
    size = states_up.shape[1]
    indices = np.linspace(0, size - 1, min(num_nodes, size), dtype=int)
    colors = []
    for idx in indices:
        line, = ax.plot(input_values, states_up[:, idx], alpha=0.6, linewidth=0.9)
        colors.append(line.get_color())
    for i, idx in enumerate(indices):
        ax.plot(input_values, states_down[:, idx], alpha=0.4, linewidth=0.7,
                linestyle="--", color=colors[i])
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Node State")
    ax.set_xlabel("Constant Input u")


def plot_rms(ax, input_values, states_up, states_down, title):
    """Plot RMS state and mean gap for sweep up/down."""
    rms_up = np.sqrt(np.mean(states_up ** 2, axis=1))
    rms_down = np.sqrt(np.mean(states_down ** 2, axis=1))
    mean_up = np.mean(states_up, axis=1)
    mean_down = np.mean(states_down, axis=1)
    mean_gap = np.abs(mean_up - mean_down)

    ax.plot(input_values, rms_up, color="C0", label="sweep up")
    ax.plot(input_values, rms_down, color="C3", label="sweep down")
    ax.plot(input_values, mean_gap, color="C4", label="mean gap")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("RMS State / Mean Gap")
    ax.set_xlabel("Constant Input u")
    ax.legend(fontsize=6)


def run_sweep_pair(mat, f, input_up, settle, seed):
    """Run both sweep-up and sweep-down, returning both state arrays."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    states_up = sweep(mat, f, input_up, settle)

    input_down = input_up[::-1].copy()
    # Reset for sweep down — rebuild same reservoir state
    f_down = Hysteresis(h_c=f.h_c, m_r=f.m_r, beta=f.beta, shift=f.shift,
                        decay_rate=f.decay_rate)
    states_down = sweep(mat, f_down, input_down, settle)
    states_down = states_down[::-1]

    return states_up, states_down


def main():
    parser = argparse.ArgumentParser(
        description="Hysteresis sweep: CMA-ES tiles vs constant vs baseline-esn",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("tile_dir", type=str,
                        help="Directory with CMA-ES tile JSONs (named *spectral_radius=X.json)")
    parser.add_argument("--rho", type=float, nargs="+", default=None,
                        help="Spectral radii to plot (default: all found in tile_dir)")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--neighborhood", type=int, default=2)
    parser.add_argument("--self_conn", type=float, default=0.0)
    parser.add_argument("--sign_frac", type=float, default=0.5)
    parser.add_argument("--dir_frac", type=float, default=0.9)
    parser.add_argument("--dir_weights", type=float, default=0.0)
    parser.add_argument("--h_c", type=float, default=0.0)
    parser.add_argument("--m_r", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=0.15)
    parser.add_argument("--decay_rate", type=float, default=2.0)
    parser.add_argument("--n_points", type=int, default=200)
    parser.add_argument("--settle", type=int, default=500)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # Find tiles
    tile_map = _find_tile_files(args.tile_dir)
    if not tile_map:
        print(f"No tile JSONs with spectral_radius=X found in {args.tile_dir}")
        return

    rho_values = args.rho if args.rho else sorted(tile_map.keys())
    rho_values = [r for r in rho_values if r in tile_map]
    if not rho_values:
        print(f"No matching spectral radii found. Available: {sorted(tile_map.keys())}")
        return

    print(f"Spectral radii: {rho_values}")
    print(f"Tiles: {[tile_map[r] for r in rho_values]}")

    row_labels = ["cmaes", "constant", "baseline-esn"]
    n_rows = len(row_labels)
    n_cols = len(rho_values)
    input_up = np.linspace(-1.0, 1.0, args.n_points)

    f_args = dict(h_c=args.h_c, m_r=args.m_r, beta=args.beta,
                  shift=args.shift, decay_rate=args.decay_rate)

    fig_nodes, ax_n = plt.subplots(n_rows, n_cols,
                                    figsize=(3.5 * n_cols, 2.5 * n_rows),
                                    squeeze=False)
    fig_rms, ax_r = plt.subplots(n_rows, n_cols,
                                  figsize=(3.5 * n_cols, 2.5 * n_rows),
                                  squeeze=False)

    for col, rho in enumerate(rho_values):
        for row, label in enumerate(row_labels):
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            if label == "cmaes":
                mat = build_reservoir_from_tile(
                    tile_map[rho], args.size, args.neighborhood,
                    args.self_conn, rho)
            else:
                mat = build_reservoir_from_type(
                    label, args.size, args.neighborhood, args.self_conn,
                    args.sign_frac, args.dir_frac, args.dir_weights, rho)

            f = Hysteresis(**f_args)
            states_up, states_down = run_sweep_pair(
                mat, f, input_up, args.settle, args.seed)

            title = f"{label}, rho={rho}"
            plot_nodes(ax_n[row, col], input_up, states_up, states_down,
                       title, args.num_nodes)
            plot_rms(ax_r[row, col], input_up, states_up, states_down, title)

            print(f"  Done: {label} @ rho={rho}")

    fig_nodes.suptitle("Node States — Hysteresis Sweep", fontsize=11)
    fig_rms.suptitle("RMS State / Mean Gap — Hysteresis Sweep", fontsize=11)
    fig_nodes.tight_layout()
    fig_rms.tight_layout()

    if args.save:
        base, ext = os.path.splitext(args.save)
        path_nodes = f"{base}_nodes{ext}"
        path_rms = f"{base}_rms{ext}"
        fig_nodes.savefig(path_nodes, dpi=args.dpi, bbox_inches="tight")
        fig_rms.savefig(path_rms, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {path_nodes} and {path_rms}")

    if not getattr(args, 'no_show', False):
        plt.show()


if __name__ == "__main__":
    main()
