import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.getcwd())
from visualization.plot_tile_hysteresis_sweep import (
    build_reservoir_from_type, build_reservoir_from_tile, run_sweep_pair, _find_tile_files,
)
from activation import ShiftedTanh

REPO = os.getcwd()
THESIS = "/Users/leonhesthaug/master/workspace/thesis"

rhos = [1.0, 1.2, 1.4]
SIZE, NH, SC = 400, 2, 0.0
SIGN, DIRF, DIRW = 0.5, 0.9, 0.0
BETA, SHIFT = 1.0, 0.15
NPTS, SETTLE, SEED, NUMNODES = 200, 500, 42, 20

TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 16, 16, 11, 13

tile_map = _find_tile_files(f"{REPO}/experiments/v2/tiles/hysteresis")
input_up = np.linspace(-1.0, 1.0, NPTS)

_cache = {}
def get_sweep(typ, rho):
    key = (typ, rho)
    if key in _cache:
        return _cache[key]
    np.random.seed(SEED); torch.manual_seed(SEED)
    if typ == "cmaes":
        mat = build_reservoir_from_tile(tile_map[rho], SIZE, NH, SC, rho)
    else:
        mat = build_reservoir_from_type(typ, SIZE, NH, SC, SIGN, DIRF, DIRW, rho)
    f = ShiftedTanh(beta=BETA, shift=SHIFT)
    su, sd = run_sweep_pair(mat, f, input_up, SETTLE, SEED)
    _cache[key] = (su, sd)
    print(f"  swept {typ} @ rho={rho}")
    return _cache[key]


def plot_nodes_cell(ax, su, sd):
    size = su.shape[1]
    idxs = np.linspace(0, size - 1, min(NUMNODES, size), dtype=int)
    colors = []
    for i in idxs:
        l, = ax.plot(input_up, su[:, i], alpha=0.6, lw=0.9)
        colors.append(l.get_color())
    for k, i in enumerate(idxs):
        ax.plot(input_up, sd[:, i], alpha=0.4, lw=0.8, ls="--", color=colors[k])
    ax.tick_params(labelsize=TICK_FS)


def plot_rms_cell(ax, su, sd):
    rms_up = np.sqrt(np.mean(su ** 2, axis=1))
    rms_dn = np.sqrt(np.mean(sd ** 2, axis=1))
    gap = np.abs(np.mean(su, axis=1) - np.mean(sd, axis=1))
    ax.plot(input_up, rms_up, color="C0", lw=1.6, label="sweep up")
    ax.plot(input_up, rms_dn, color="C3", lw=1.6, label="sweep down")
    ax.plot(input_up, gap, color="C4", lw=1.5, label="mean gap")
    ax.tick_params(labelsize=TICK_FS)


def make_fig(cols, kind, supy, out):
    fig, axes = plt.subplots(3, 2, figsize=(9, 10.5), sharex=True, constrained_layout=True)
    for r, rho in enumerate(rhos):
        for c, (name, typ) in enumerate(cols):
            su, sd = get_sweep(typ, rho)
            ax = axes[r][c]
            (plot_nodes_cell if kind == "nodes" else plot_rms_cell)(ax, su, sd)
            if r == 0:
                ax.set_title(name, fontsize=TITLE_FS)
            if c == len(cols) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(rf"$\rho = {rho}$", fontsize=LABEL_FS,
                              rotation=270, labelpad=20)
    fig.supxlabel("Constant input $u$", fontsize=LABEL_FS)
    fig.supylabel(supy, fontsize=LABEL_FS)

    if kind == "nodes":
        handles = [Line2D([0], [0], color="0.3", lw=1.6, label="sweep up"),
                   Line2D([0], [0], color="0.3", lw=1.6, ls="--", label="sweep down")]
    else:
        handles = [Line2D([0], [0], color="C0", lw=1.8, label="sweep up"),
                   Line2D([0], [0], color="C3", lw=1.8, label="sweep down"),
                   Line2D([0], [0], color="C4", lw=1.6, label="mean gap")]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles),
               fontsize=LEG_FS, frameon=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# 5.3.4 : constant lattice + baseline ESN
sc534 = [("Constant lattice", "constant"), ("Baseline ESN", "baseline-esn")]
D534 = f"{THESIS}/Figures/experiments/self_connections_and_hysteresis/hysteresis"
make_fig(sc534, "nodes", "Node state", "/tmp/s534_nodes.png")
make_fig(sc534, "rms", "RMS state / mean gap", "/tmp/s534_rms.png")

# 6.3.6 : CMA-ES tile + constant lattice
sc636 = [("CMA-ES tile", "cmaes"), ("Constant lattice", "constant")]
make_fig(sc636, "nodes", "Node state", "/tmp/s636_nodes.png")
make_fig(sc636, "rms", "RMS state / mean gap", "/tmp/s636_rms.png")
print("DONE")
