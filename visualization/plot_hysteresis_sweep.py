"""Hysteresis sweep analysis for ESN reservoirs.

Two modes:
  steady-state (default): Settles at each input level, compares sweep-up vs
      sweep-down equilibria. Tests for rate-independent hysteresis.
  continuous (--continuous): Feeds a triangular wave at different speeds,
      plots input-vs-state loops. Tests for rate-dependent hysteresis.

Usage:
    # Rate-independent (steady-state)
    python -m visualization.plot_hysteresis_sweep
    python -m visualization.plot_hysteresis_sweep --rho 0.95 1.1 --self_conn 0 1

    # Rate-dependent (continuous)
    python -m visualization.plot_hysteresis_sweep --continuous
    python -m visualization.plot_hysteresis_sweep --continuous --speeds 50 200 1000
    python -m visualization.plot_hysteresis_sweep --continuous --self_conn 0 1 3 5

    # Common options
    python -m visualization.plot_hysteresis_sweep --h_c 0.3 --type baseline-esn
    python -m visualization.plot_hysteresis_sweep --save sweep.png --dpi 300
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from matrix import Matrix
from activation import Hysteresis


def build_reservoir(size, res_type, neighborhood, self_conn, sign_frac,
                    dir_frac, dir_weights, rho, h_c, m_r, beta, shift,
                    decay_rate):
    """Build a reservoir and activation function with the given parameters."""
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

    # Rescale to target spectral radius
    if rho is not None:
        spec_rad = float(torch.max(torch.abs(torch.linalg.eigvals(mat.W_res))).real)
        if spec_rad > 0:
            mat.W_res *= rho / spec_rad

    f = Hysteresis(h_c=h_c, m_r=m_r, beta=beta, shift=shift, decay_rate=decay_rate)
    return mat, f


# ---------- Steady-state (rate-independent) mode ----------

def sweep(mat, f, input_values, settle_steps):
    """Run the reservoir to steady state at each constant input value.

    Returns:
        states: (len(input_values), size) array of final node states
    """
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


def plot_sweep(axes_top, axes_bot, input_values, states_up, states_down,
               title, num_nodes_plot=20):
    """Plot node states (top) and RMS/gap (bottom) for one parameter combo."""
    n_inputs, size = states_up.shape

    # Top: individual node activations
    ax = axes_top
    indices = np.linspace(0, size - 1, min(num_nodes_plot, size), dtype=int)
    for idx in indices:
        ax.plot(input_values, states_up[:, idx], alpha=0.5, linewidth=0.7)
    # Overlay sweep-down in lighter color
    for idx in indices:
        ax.plot(input_values, states_down[:, idx], alpha=0.3, linewidth=0.5,
                linestyle="--")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Node State")
    ax.set_xlabel("Constant Input u")
    ax.legend(["sweep up", "sweep down"], fontsize=6, loc="upper left")

    # Bottom: RMS state and mean gap
    ax = axes_bot
    rms_up = np.sqrt(np.mean(states_up ** 2, axis=1))
    rms_down = np.sqrt(np.mean(states_down ** 2, axis=1))
    mean_up = np.mean(states_up, axis=1)
    mean_down = np.mean(states_down, axis=1)
    mean_gap = np.abs(mean_up - mean_down)

    ax.plot(input_values, rms_up, color="C0", label="sweep up (RMS)")
    ax.plot(input_values, rms_down, color="C3", label="sweep down (RMS)")
    ax.plot(input_values, mean_gap, color="C4", label="mean gap")
    ax.set_ylabel("RMS State / Mean Gap")
    ax.set_xlabel("Constant Input u")
    ax.legend(fontsize=6, loc="upper right")


def run_steady_state(args):
    """Run the steady-state (rate-independent) sweep.

    Returns a list of (fig, suffix) tuples. When multiple neighborhoods are
    given, node-state and RMS/gap plots are split into separate figures.
    """
    input_up = np.linspace(args.input_range[0], args.input_range[1], args.n_points)
    input_down = input_up[::-1].copy()

    n_neighborhoods = len(args.neighborhood)
    n_cols = len(args.rho) * len(args.self_conn)
    multi_nh = n_neighborhoods > 1

    if multi_nh:
        fig_nodes, ax_nodes = plt.subplots(
            n_neighborhoods, n_cols,
            figsize=(3.5 * n_cols, 2.5 * n_neighborhoods), squeeze=False,
        )
        fig_rms, ax_rms = plt.subplots(
            n_neighborhoods, n_cols,
            figsize=(3.5 * n_cols, 2.5 * n_neighborhoods), squeeze=False,
        )
    else:
        fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), squeeze=False)

    for row, nh in enumerate(args.neighborhood):
        col = 0
        for rho in args.rho:
            for sc in args.self_conn:
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                mat, f = build_reservoir(
                    size=args.size, res_type=args.type,
                    neighborhood=nh, self_conn=sc,
                    sign_frac=args.sign_frac, dir_frac=args.dir_frac,
                    dir_weights=args.dir_weights, rho=rho,
                    h_c=args.h_c, m_r=args.m_r, beta=args.beta,
                    shift=args.shift, decay_rate=args.decay_rate,
                )
                states_up = sweep(mat, f, input_up, args.settle)

                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                mat2, f2 = build_reservoir(
                    size=args.size, res_type=args.type,
                    neighborhood=nh, self_conn=sc,
                    sign_frac=args.sign_frac, dir_frac=args.dir_frac,
                    dir_weights=args.dir_weights, rho=rho,
                    h_c=args.h_c, m_r=args.m_r, beta=args.beta,
                    shift=args.shift, decay_rate=args.decay_rate,
                )
                states_down = sweep(mat2, f2, input_down, args.settle)
                states_down = states_down[::-1]

                if multi_nh:
                    title = f"NH={nh}, rho={rho}" + (f", w_self={sc}" if len(args.self_conn) > 1 else "")
                    plot_sweep(ax_nodes[row, col], ax_rms[row, col],
                               input_values=input_up, states_up=states_up,
                               states_down=states_down, title=title,
                               num_nodes_plot=args.num_nodes_plot)
                else:
                    title = f"rho={rho}, w_self={sc}"
                    plot_sweep(axes[0, col], axes[1, col], input_up, states_up,
                               states_down, title, num_nodes_plot=args.num_nodes_plot)
                col += 1

        if multi_nh:
            print(f"  Completed neighborhood {nh}")

    suptitle_base = (
        f"type={args.type}, "
        f"h_c={args.h_c}, m_r={args.m_r}, beta={args.beta}, "
        f"shift={args.shift}, decay={args.decay_rate}"
    )

    if multi_nh:
        fig_nodes.suptitle(f"Node States — {suptitle_base}", fontsize=10)
        fig_rms.suptitle(f"RMS State / Mean Gap — {suptitle_base}", fontsize=10)
        fig_nodes.tight_layout()
        fig_rms.tight_layout()
        return [(fig_nodes, "_nodes"), (fig_rms, "_rms")]
    else:
        nh = args.neighborhood[0]
        fig.suptitle(f"Rate-Independent Sweep — NH={nh}, {suptitle_base}", fontsize=10)
        fig.tight_layout()
        return [(fig, "")]


# ---------- Continuous (rate-dependent) mode ----------

def continuous_sweep(mat, f, input_signal):
    """Drive reservoir with a continuous input signal, recording state at every step.

    Returns:
        states: (len(input_signal), size) array of node states
    """
    size = mat.W_res.shape[0]
    x = torch.zeros(size)
    f.reset(size)
    states = np.zeros((len(input_signal), size))

    for t, u_val in enumerate(input_signal):
        u = torch.tensor(u_val, dtype=torch.float32)
        state_input = mat.W_in * u + mat.W_res.mv(x)
        x = f(state_input)
        states[t] = x.detach().numpy()

    return states


def make_triangle_wave(amplitude, period_steps, num_cycles, warmup_cycles=2):
    """Generate a triangular wave: ramps up then down repeatedly.

    Returns the full signal including warmup, and the index where
    the final cycle starts (for plotting only the steady-state cycle).
    """
    half = period_steps // 2
    up = np.linspace(-amplitude, amplitude, half, endpoint=False)
    down = np.linspace(amplitude, -amplitude, period_steps - half, endpoint=False)
    one_cycle = np.concatenate([up, down])
    total_cycles = warmup_cycles + num_cycles
    signal = np.tile(one_cycle, total_cycles)
    plot_start = warmup_cycles * period_steps
    return signal, plot_start


def plot_continuous(axes_top, axes_bot, input_signal, states, plot_start,
                    title, speeds, speed_idx, num_nodes_plot=20):
    """Plot input-vs-mean-state loop (top) and loop area vs speed (bottom)."""
    size = states.shape[1]
    u_plot = input_signal[plot_start:]
    s_plot = states[plot_start:]

    mean_state = np.mean(s_plot, axis=1)
    rms_state = np.sqrt(np.mean(s_plot ** 2, axis=1))

    # Top: input vs mean state (hysteresis loop)
    ax = axes_top
    ax.plot(u_plot, mean_state, alpha=0.8, linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Mean Node State")
    ax.set_xlabel("Input u")

    # Bottom: input vs RMS state
    ax = axes_bot
    ax.plot(u_plot, rms_state, alpha=0.8, linewidth=0.8)
    ax.set_ylabel("RMS State")
    ax.set_xlabel("Input u")


def run_continuous(args):
    """Run the continuous (rate-dependent) sweep.

    Returns a list of (fig, suffix) tuples.
    """
    n_params = len(args.rho) * len(args.self_conn)
    n_speeds = len(args.speeds)
    n_cols = n_params * n_speeds

    n_neighborhoods = len(args.neighborhood)
    multi_nh = n_neighborhoods > 1

    if multi_nh:
        fig_mean, ax_mean = plt.subplots(
            n_neighborhoods, n_cols,
            figsize=(3.5 * n_cols, 2.5 * n_neighborhoods), squeeze=False,
        )
        fig_rms, ax_rms = plt.subplots(
            n_neighborhoods, n_cols,
            figsize=(3.5 * n_cols, 2.5 * n_neighborhoods), squeeze=False,
        )
    else:
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)

    for row, nh in enumerate(args.neighborhood):
        col = 0
        for rho in args.rho:
            for sc in args.self_conn:
                for si, speed in enumerate(args.speeds):
                    np.random.seed(args.seed)
                    torch.manual_seed(args.seed)
                    mat, f = build_reservoir(
                        size=args.size, res_type=args.type,
                        neighborhood=nh, self_conn=sc,
                        sign_frac=args.sign_frac, dir_frac=args.dir_frac,
                        dir_weights=args.dir_weights, rho=rho,
                        h_c=args.h_c, m_r=args.m_r, beta=args.beta,
                        shift=args.shift, decay_rate=args.decay_rate,
                    )

                    signal, plot_start = make_triangle_wave(
                        amplitude=args.input_range[1],
                        period_steps=speed,
                        num_cycles=args.cycles,
                        warmup_cycles=args.warmup_cycles,
                    )
                    states = continuous_sweep(mat, f, signal)

                    if multi_nh:
                        title = f"NH={nh}, rho={rho}" + (f", w_self={sc}" if len(args.self_conn) > 1 else "") + f"\nperiod={speed}"
                        plot_continuous(ax_mean[row, col], ax_rms[row, col],
                                        signal, states, plot_start, title,
                                        args.speeds, si,
                                        num_nodes_plot=args.num_nodes_plot)
                    else:
                        title = f"rho={rho}, w_self={sc}\nperiod={speed}"
                        plot_continuous(axes[0, col], axes[1, col], signal, states,
                                        plot_start, title, args.speeds, si,
                                        num_nodes_plot=args.num_nodes_plot)
                    col += 1

        if multi_nh:
            print(f"  Completed neighborhood {nh}")

    suptitle_base = (
        f"type={args.type}, "
        f"h_c={args.h_c}, m_r={args.m_r}, beta={args.beta}, "
        f"shift={args.shift}, decay={args.decay_rate}"
    )

    if multi_nh:
        fig_mean.suptitle(f"Mean State Loops — {suptitle_base}", fontsize=10)
        fig_rms.suptitle(f"RMS State Loops — {suptitle_base}", fontsize=10)
        fig_mean.tight_layout()
        fig_rms.tight_layout()
        return [(fig_mean, "_mean"), (fig_rms, "_rms")]
    else:
        nh = args.neighborhood[0]
        fig.suptitle(
            f"Rate-Dependent Sweep — NH={nh}, {suptitle_base}",
            fontsize=10,
        )
        fig.tight_layout()
        return [(fig, "")]


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Hysteresis sweep analysis for ESN reservoirs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    parser.add_argument("--continuous", action="store_true",
                        help="Rate-dependent mode: triangular wave at different speeds")

    # Sweep grid parameters (each list creates subplot columns)
    parser.add_argument("--rho", type=float, nargs="+", default=[0.95],
                        help="Spectral radii to sweep")
    parser.add_argument("--self_conn", type=float, nargs="+", default=[0.0],
                        help="Self-connection weights to sweep")

    # Reservoir structure
    parser.add_argument("--size", type=int, default=400,
                        help="Reservoir size (must be perfect square for lattice)")
    parser.add_argument("--type", type=str, default="constant",
                        help="Reservoir type: constant, random, baseline-esn, orthogonal")
    parser.add_argument("--neighborhood", type=int, nargs="+", default=[2],
                        help="Neighborhood shells (1=Von Neumann, 2=Moore, ...)")
    parser.add_argument("--sign_frac", type=float, default=0.5,
                        help="Fraction of edges made negative")
    parser.add_argument("--dir_frac", type=float, default=0.9,
                        help="Fraction of edge pairs made unidirectional")
    parser.add_argument("--dir_weights", type=float, default=0.0,
                        help="Weight multiplier for removed direction")

    # Activation / hysteresis
    parser.add_argument("--h_c", type=float, default=0.0,
                        help="Coercivity (0 = standard tanh, >0 = hysteresis)")
    parser.add_argument("--m_r", type=float, default=1.0,
                        help="Remanence scaling factor")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Steepness of tanh")
    parser.add_argument("--shift", type=float, default=0.15,
                        help="Horizontal shift of tanh")
    parser.add_argument("--decay_rate", type=float, default=2.0,
                        help="Minor loop merge speed")

    # Steady-state mode settings
    parser.add_argument("--n_points", type=int, default=200,
                        help="Number of input values in steady-state sweep")
    parser.add_argument("--settle", type=int, default=500,
                        help="Timesteps to settle at each input level")

    # Continuous mode settings
    parser.add_argument("--speeds", type=int, nargs="+", default=[20, 100, 500],
                        help="Triangle wave period in timesteps (smaller = faster)")
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of cycles to plot (after warmup)")
    parser.add_argument("--warmup_cycles", type=int, default=2,
                        help="Warmup cycles before plotting")

    # Shared settings
    parser.add_argument("--input_range", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Input sweep range [min, max]")
    parser.add_argument("--num_nodes_plot", type=int, default=20,
                        help="Number of nodes to show in node-state plots")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Output
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to path (e.g. sweep.png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution for saved figure")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display the figure")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.continuous:
        figures = run_continuous(args)
    else:
        figures = run_steady_state(args)

    if args.save:
        import os
        base, ext = os.path.splitext(args.save)
        for fig, suffix in figures:
            path = f"{base}{suffix}{ext}"
            fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved to {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
