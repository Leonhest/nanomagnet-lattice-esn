"""Visualize the alternating projections algorithm for thesis figures.

Three panels:
  1. Initial sparse lattice W_res (adjacency matrix as heatmap)
  2. After SVD orthogonal projection (dense orthogonal matrix)
  3. After sparsity projection (re-zeroed to lattice pattern)

Shows one iteration of the algorithm. Optionally shows multiple iterations.

Usage:
    python -m visualization.plot_alternating_projections
    python -m visualization.plot_alternating_projections --size 100 --neighborhood 2
    python -m visualization.plot_alternating_projections --iterations 5 --save alt_proj.png
"""

import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from matrix import Matrix
from utils.formula import orthogonality_error


def build_lattice_matrix(size, neighborhood, sign_frac, dir_frac, dir_weights,
                         self_conn, seed):
    """Build a lattice reservoir matrix and return it as a numpy array."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    W_args = {
        "size": size,
        "W_in_args": {"input_scale": 0.1, "distribution": "fixed"},
        "W_res_args": {
            "type": "constant",
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
    return mat.W_res.numpy()


def sparsity(W):
    """Fraction of zero entries."""
    return 1.0 - np.count_nonzero(W) / W.size


def plot_matrix_panel(ax, W, title, vmin, vmax, cmap="RdBu_r"):
    """Plot a matrix as a heatmap on the given axes."""
    im = ax.imshow(W, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Node index")
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Visualize alternating projections algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, default=100,
                        help="Reservoir size (perfect square)")
    parser.add_argument("--neighborhood", type=int, default=2,
                        help="Neighborhood shells")
    parser.add_argument("--sign_frac", type=float, default=0.5)
    parser.add_argument("--dir_frac", type=float, default=0.9)
    parser.add_argument("--dir_weights", type=float, default=0.0)
    parser.add_argument("--self_conn", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of alternating projection iterations to show")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    # Build initial matrix
    W_init = build_lattice_matrix(
        args.size, args.neighborhood, args.sign_frac, args.dir_frac,
        args.dir_weights, args.self_conn, args.seed,
    )
    mask = (W_init != 0)

    # Shared color scale
    vmax = max(abs(W_init.min()), abs(W_init.max()), 1.0)

    if args.iterations == 1:
        # Three-panel figure: initial → orthogonal → sparse
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        # Panel 1: Initial sparse matrix
        orth_err = orthogonality_error(W_init)
        spar = sparsity(W_init)
        plot_matrix_panel(axes[0], W_init,
                          "1. Initial lattice matrix",
                          -vmax, vmax)
        axes[0].set_ylabel("Node index")
        axes[0].text(0.5, -0.22, f"Sparsity: {spar:.1%} | Orth. error: {orth_err:.3f}",
                     transform=axes[0].transAxes, ha="center", fontsize=8)

        # Panel 2: After SVD orthogonal projection (dense)
        U, _, Vh = np.linalg.svd(W_init, full_matrices=True)
        W_ortho = (U @ Vh).astype(np.float32)
        orth_err = orthogonality_error(W_ortho)
        spar = sparsity(W_ortho)
        plot_matrix_panel(axes[1], W_ortho,
                          "2. Orthogonal projection (SVD)",
                          -vmax, vmax)
        axes[1].set_yticks([])
        axes[1].text(0.5, -0.22, f"Sparsity: {spar:.1%} | Orth. error: {orth_err:.3f}",
                     transform=axes[1].transAxes, ha="center", fontsize=8)

        # Panel 3: After sparsity projection (re-zero)
        W_sparse = W_ortho * mask
        orth_err = orthogonality_error(W_sparse)
        spar = sparsity(W_sparse)
        im = plot_matrix_panel(axes[2], W_sparse,
                               "3. Sparsity projection (re-zero)",
                               -vmax, vmax)
        axes[2].set_yticks([])
        axes[2].text(0.5, -0.22, f"Sparsity: {spar:.1%} | Orth. error: {orth_err:.3f}",
                     transform=axes[2].transAxes, ha="center", fontsize=8)

        # Add arrows between panels (shifted left to avoid next panel's y-axis)
        for i in range(2):
            gap_left = axes[i].get_position().x1
            gap_right = axes[i+1].get_position().x0
            mid_x = gap_left + (gap_right - gap_left) * 0.6
            fig.text(mid_x, 0.5, "→", fontsize=24,
                     ha="center", va="center", fontweight="bold")

        # Colorbar
        fig.subplots_adjust(right=0.91, wspace=0.15)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Weight")

    else:
        # Multi-iteration: show initial + each iteration's two steps
        n_cols = 1 + 2 * args.iterations
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

        # Initial
        orth_err = orthogonality_error(W_init)
        spar = sparsity(W_init)
        plot_matrix_panel(axes[0], W_init, "Initial", -vmax, vmax)
        axes[0].text(0.5, -0.15, f"Spar: {spar:.1%}\nOrth: {orth_err:.3f}",
                     transform=axes[0].transAxes, ha="center", fontsize=7)

        W = W_init.copy()
        col = 1
        for it in range(args.iterations):
            # Orthogonal projection
            U, _, Vh = np.linalg.svd(W, full_matrices=True)
            W_ortho = (U @ Vh).astype(np.float32)
            orth_err = orthogonality_error(W_ortho)
            spar = sparsity(W_ortho)
            plot_matrix_panel(axes[col], W_ortho,
                              f"Iter {it+1}: SVD", -vmax, vmax)
            axes[col].text(0.5, -0.15, f"Spar: {spar:.1%}\nOrth: {orth_err:.3f}",
                           transform=axes[col].transAxes, ha="center", fontsize=7)
            col += 1

            # Sparsity projection
            W = W_ortho * mask
            orth_err = orthogonality_error(W)
            spar = sparsity(W)
            im = plot_matrix_panel(axes[col], W,
                                   f"Iter {it+1}: Sparse", -vmax, vmax)
            axes[col].text(0.5, -0.15, f"Spar: {spar:.1%}\nOrth: {orth_err:.3f}",
                           transform=axes[col].transAxes, ha="center", fontsize=7)
            col += 1

        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Weight")

    fig.suptitle("Alternating Projections: Orthogonality ↔ Sparsity",
                 fontsize=13, fontweight="bold")

    if args.save:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved to {args.save}")

    if not getattr(args, 'no_show', False):
        plt.show()


if __name__ == "__main__":
    main()
