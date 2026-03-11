"""
Thesis-quality 2-panel figure illustrating a lattice reservoir with self-connections
and the resulting hysteresis in the activation function.

(a) 6×6 Moore-neighborhood lattice with self-loops highlighted
(b) Activation function with hysteresis loop from self-connection feedback

The ESN state update: x[t] = tanh(β * (W_in * u[t] + W_res @ x[t-1]) - shift)
Self-connections (diagonal of W_res) feed x[t-1] back, creating hysteresis.

Usage:
    python -m utils.plot_lattice_hysteresis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUTPUT_PATH = "experiments/conference/methods/lattice_hysteresis_illustration.png"
LATTICE_DIM = 4


def _grid_pos(node):
    return (node[1], -node[0])


def panel_a_lattice(ax):
    """Panel (a): 4×4 Moore lattice with self-loops visualized."""
    m = LATTICE_DIM

    # Build Moore-neighborhood graph (8-connectivity)
    G = nx.grid_graph([m, m])
    for i in range(m):
        for j in range(m):
            for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < m:
                    G.add_edge((i, j), (ni, nj))
    G = G.to_directed()

    # Add self-connections with a distinct weight
    for node in list(G.nodes()):
        G.add_edge(node, node, weight=1.0)
    # Set neighbor weights
    for u, v, d in G.edges(data=True):
        if u != v:
            d["weight"] = 0.5

    pos = {nd: (nd[1], -nd[0]) for nd in G.nodes()}

    # Use the same rendering approach as plot_lattice in matrix.py
    # Color edges by weight — self-loops (1.0) vs neighbors (0.5)
    edge_colors = ["#d94a4a" if u == v else "#888888" for u, v in G.edges()]
    edge_widths = [2.0 if u == v else 0.8 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=350,
                           node_color="#4a90d9", edgecolors="black", linewidths=1.2)
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_colors, width=edge_widths,
        connectionstyle="arc3,rad=0.15", arrows=True, arrowsize=10,
    )

    ax.set_title("(a) Lattice with self-connections", fontsize=13, fontweight="bold", pad=10)
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


def panel_b_hysteresis(ax):
    """Panel (b): Hysteresis loop from self-connection in tanh activation.

    Steady state: x* = tanh(β * (I + w_self * x*) - shift)
    Sweeping input I up then down traces different paths when w_self is large.
    """
    beta = 1.5
    shift = 0.0
    w_self = 1.0

    I_range = np.linspace(-4, 4, 800)

    def find_steady_state(I_val, x_init, n_iter=200):
        x = x_init
        for _ in range(n_iter):
            x = np.tanh(beta * (I_val + w_self * x) - shift)
        return x

    # Sweep input up (starting from low state)
    x_up = np.zeros_like(I_range)
    x_prev = -1.0
    for i, I_val in enumerate(I_range):
        x_prev = find_steady_state(I_val, x_prev)
        x_up[i] = x_prev

    # Sweep input down (starting from high state)
    x_down = np.zeros_like(I_range)
    x_prev = 1.0
    for i, I_val in enumerate(I_range[::-1]):
        x_prev = find_steady_state(I_val, x_prev)
        x_down[len(I_range) - 1 - i] = x_prev

    # Also show no self-connection for comparison
    x_no_self = np.tanh(beta * I_range - shift)

    ax.plot(I_range, x_no_self, '--', color='#888888', linewidth=1.5,
            label=r'No self-conn ($w_{self}=0$)', zorder=2)
    ax.plot(I_range, x_up, '-', color='#d94a4a', linewidth=2.2,
            label=r'Sweep input $\uparrow$', zorder=3)
    ax.plot(I_range, x_down, '-', color='#4a90d9', linewidth=2.2,
            label=r'Sweep input $\downarrow$', zorder=3)

    # Shade the hysteresis region
    # Find where the curves diverge
    diff = np.abs(x_up - x_down)
    hysteresis_mask = diff > 0.05
    if np.any(hysteresis_mask):
        ax.fill_between(I_range, x_up, x_down, where=hysteresis_mask,
                         alpha=0.12, color='#9b59b6', zorder=1)

    ax.set_xlabel(r'Net input $I = W_{in} \cdot u[t] + \sum_{j \neq i} W_{ij} x_j[t-1]$', fontsize=10)
    ax.set_ylabel(r'Node state $x_i[t]$', fontsize=10)
    ax.set_title(rf'(b) Hysteresis ($w_{{self}}={w_self}$, $\beta={beta}$)',
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.5, zorder=0)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.15, 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)


def panel_c_fixed_point(ax):
    """Panel (c): Fixed-point diagram showing y=x vs y=tanh(β(I + w_self·x)) intersections."""
    beta = 1.5
    shift = 0.0
    w_self = 1.0

    x_range = np.linspace(-2, 2, 500)

    # y = x line
    ax.plot(x_range, x_range, 'k-', linewidth=2, label=r'$y = x$')

    # tanh curves for several I values
    I_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(I_values)))
    for I_val, color in zip(I_values, colors):
        y = np.tanh(beta * (I_val + w_self * x_range) - shift)
        ax.plot(x_range, y, '-', color=color, linewidth=1.8,
                label=rf'$I = {I_val:+.1f}$')

        # Mark intersection(s): where y = x meets y = tanh(...)
        diff = x_range - np.tanh(beta * (I_val + w_self * x_range) - shift)
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        for idx in sign_changes:
            # Linear interpolation for intersection x
            x_int = x_range[idx] - diff[idx] * (x_range[idx+1] - x_range[idx]) / (diff[idx+1] - diff[idx])
            ax.plot(x_int, x_int, 'o', color=color, markersize=8, markeredgecolor='black',
                    markeredgewidth=0.8, zorder=5)

    ax.set_xlabel(r'$x^*$', fontsize=12)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.set_title(rf'(c) Fixed-point diagram ($w_{{self}}={w_self}$, $\beta={beta}$)',
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.axhline(0, color='gray', linewidth=0.5, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.5, zorder=0)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)


def main():
    global n
    n = LATTICE_DIM

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    panel_a_lattice(axes[0])
    panel_b_hysteresis(axes[1])
    panel_c_fixed_point(axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
