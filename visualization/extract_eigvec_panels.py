"""Extract individual panels from eigvec_viz HTML files as static images.

Usage:
    python -m visualization.extract_eigvec_panels <eigvec.html> [options]

Options:
    --eigvec-idx=N      Eigenvalue index for dynamic panels (default: 0)
    --output-dir=DIR    Output directory (default: <html_stem>_panels/)
    --format=FMT        Output format: pdf, png, svg (default: pdf)
    --panels=PANELS     Comma-separated panel names to extract (default: all)
    --dpi=N             DPI for raster formats (default: 300)

Available panels:
    spectrum, magnitude, phase, local_r, total_activity, total_local_r,
    resolvent_z1, resolvent_z_complex_mag, resolvent_z_complex_phase,
    resolvent_local_r, resolvent_diag, relative_self_influence,
    sv_spectrum, sv_distribution, resolvent_phase_flow, phase_flow
"""

import json
import math
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_eigvec_data(html_path):
    """Extract the eigvec-data JSON blob from an eigvec_viz HTML file."""
    with open(html_path, "r") as f:
        content = f.read()

    match = re.search(
        r'<script\s+type="application/json"\s+id="eigvec-data">\s*(.*?)\s*</script>',
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find eigvec-data JSON in HTML file")

    return json.loads(match.group(1))


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

_PHASE_COLORS = [
    (225, 216, 226), (165, 191, 202), (109, 144, 191), (94, 87, 176),
    (83, 29, 124), (47, 20, 54), (99, 24, 75), (158, 59, 79),
    (192, 116, 93), (208, 178, 158), (225, 216, 225),
]

def _phase_cmap():
    """Cyclic phase colormap matching the Plotly version in eigvec_viz."""
    colors = [(r / 255, g / 255, b / 255) for r, g, b in _PHASE_COLORS]
    return LinearSegmentedColormap.from_list("phase_cyclic", colors)


# ---------------------------------------------------------------------------
# Recompute derived data from W_res
# ---------------------------------------------------------------------------

def _recompute_resolvent(W_res, m):
    """Recompute all resolvent-derived grids from the weight matrix."""
    n = W_res.shape[0]
    I = np.eye(n)

    # (I - W)^{-1} · 1
    try:
        res_inv = np.linalg.inv(I - W_res)
        res_z1 = (res_inv @ np.ones(n)).reshape(m, m)
    except np.linalg.LinAlgError:
        res_inv = np.full((n, n), np.nan)
        res_z1 = np.zeros((m, m))

    # Complex resolvent: (e^{i*0.1}I - W)^{-1} · 1
    z = np.exp(1j * 0.1)
    try:
        res_z_complex = np.linalg.inv(z * I - W_res) @ np.ones(n)
        res_z_mag = np.abs(res_z_complex).reshape(m, m)
        res_z_phase = np.angle(res_z_complex).reshape(m, m)
    except np.linalg.LinAlgError:
        res_z_mag = np.zeros((m, m))
        res_z_phase = np.zeros((m, m))
        res_z_complex = np.zeros(n)

    # Local phase coherence of resolvent
    adj_threshold = 1e-8 * np.max(np.abs(W_res))
    adj = (np.abs(W_res) > adj_threshold) | (np.abs(W_res.T) > adj_threshold)
    np.fill_diagonal(adj, False)
    deg_plus1 = adj.sum(axis=1).astype(float) + 1.0

    phases = np.angle(res_z_complex)
    phasors = np.exp(1j * phases)
    nbr_sum = adj @ phasors + phasors
    res_local_r = (np.abs(nbr_sum) / deg_plus1).reshape(m, m)

    # diag((I - W)^{-1})
    res_diag = np.real(np.diag(res_inv)).reshape(m, m)

    # Relative self-influence: |R_jj| / Σ_k |R_jk|
    res_diag_abs = np.abs(np.diag(res_inv))
    res_row_sums = np.sum(np.abs(res_inv), axis=1)
    res_rel_self = (res_diag_abs / res_row_sums).reshape(m, m)

    # Singular values
    sing_vals = np.linalg.svd(W_res, compute_uv=False)

    return {
        "res_z1": res_z1,
        "res_z_mag": res_z_mag,
        "res_z_phase": res_z_phase,
        "res_local_r": res_local_r,
        "res_diag": res_diag,
        "res_rel_self": res_rel_self,
        "sing_vals": sing_vals,
    }


# ---------------------------------------------------------------------------
# Phase flow (quiver) computation
# ---------------------------------------------------------------------------

def _compute_phase_flow(phase_grid):
    """Compute phase gradient field for quiver plot.

    Returns U, V arrays where (U[r,c], V[r,c]) is the local phase gradient.
    """
    m = len(phase_grid)
    phase = np.array(phase_grid)
    U = np.zeros((m, m))
    V = np.zeros((m, m))

    for r in range(m):
        for c in range(m):
            if c < m - 1:
                dp = phase[r, c + 1] - phase[r, c]
                U[r, c] = -math.atan2(math.sin(dp), math.cos(dp)) / math.pi
            if r < m - 1:
                dp = phase[r + 1, c] - phase[r, c]
                V[r, c] = -math.atan2(math.sin(dp), math.cos(dp)) / math.pi

    return U, V


def _plot_quiver(ax, phase_grid, title, m):
    """Plot a phase flow quiver field with HSV-colored arrows."""
    U, V = _compute_phase_flow(phase_grid)
    mag = np.sqrt(U ** 2 + V ** 2)
    # Direction angle mapped to hue
    angle = np.arctan2(V, U)
    hue = (angle + np.pi) / (2 * np.pi)  # [0, 1]

    cols = np.arange(m)
    rows = np.arange(m)
    C, R = np.meshgrid(cols, rows)

    # Color by direction angle using HSV
    from matplotlib.colors import hsv_to_rgb
    hsv = np.stack([hue, np.ones_like(hue), 0.9 * np.ones_like(hue)], axis=-1)
    colors = hsv_to_rgb(hsv).reshape(-1, 3)

    # Mask out near-zero arrows
    mask = mag.ravel() > 1e-10
    ax.quiver(
        C.ravel()[mask], R.ravel()[mask],
        U.ravel()[mask], -V.ravel()[mask],  # flip V for image coords
        color=colors[mask],
        scale=25, width=0.004, headwidth=3.5, headlength=4,
    )
    ax.set_xlim(-0.5, m - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


# ---------------------------------------------------------------------------
# Individual panel plotters
# ---------------------------------------------------------------------------

def _plot_heatmap(ax, grid, title, cmap="viridis", vmin=None, vmax=None,
                  cbar_label="", center_zero=False):
    """Plot a single heatmap panel."""
    grid = np.asarray(grid)
    if center_zero:
        absmax = np.nanmax(np.abs(grid))
        vmin, vmax = -absmax, absmax
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8, pad=0.04)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")


ALL_PANELS = [
    "spectrum", "magnitude", "phase", "local_r",
    "total_activity", "total_local_r",
    "resolvent_z1", "resolvent_z_complex_mag", "resolvent_z_complex_phase",
    "resolvent_local_r", "resolvent_diag", "relative_self_influence",
    "sv_spectrum", "sv_distribution",
    "resolvent_phase_flow", "phase_flow",
]


def extract_panels(html_path, *, eigvec_idx=0, output_dir=None, fmt="pdf",
                   panels=None, dpi=300):
    """Extract panels from an eigvec_viz HTML file as individual images.

    Parameters
    ----------
    html_path : str
        Path to the eigvec HTML file.
    eigvec_idx : int
        Eigenvalue index for dynamic panels (magnitude, phase, local_r, phase_flow).
    output_dir : str or None
        Output directory. Defaults to ``<html_stem>_panels/``.
    fmt : str
        Image format: 'pdf', 'png', or 'svg'.
    panels : list of str or None
        Which panels to extract. None = all.
    dpi : int
        DPI for raster output.
    """
    print(f"Loading data from {html_path}...")
    data = extract_eigvec_data(html_path)

    m = data["m"]
    n = len(data["eigvals_re"])
    eigvals = np.array(data["eigvals_re"]) + 1j * np.array(data["eigvals_im"])
    eigvals_abs = np.array(data["eigvals_abs"])
    kuramoto = np.array(data["kuramoto"])
    sr = float(eigvals_abs[0])

    W_res = np.array(data["W_res_original"])

    if output_dir is None:
        output_dir = os.path.splitext(html_path)[0] + "_panels"
    os.makedirs(output_dir, exist_ok=True)

    if panels is None:
        panels = ALL_PANELS

    phase_cmap = _phase_cmap()

    # Recompute resolvent data if any resolvent/SV panels requested
    resolvent_panels = {
        "resolvent_z1", "resolvent_z_complex_mag", "resolvent_z_complex_phase",
        "resolvent_local_r", "resolvent_diag", "relative_self_influence",
        "sv_spectrum", "sv_distribution", "resolvent_phase_flow",
    }
    res = None
    if resolvent_panels & set(panels):
        print("Recomputing resolvent data from W_res...")
        res = _recompute_resolvent(W_res, m)

    saved = []

    def _save(fig, name):
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"  Saved {path}")

    # --- Eigenvalue Spectrum ---
    if "spectrum" in panels:
        fig, ax = plt.subplots(figsize=(6, 6))
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="gray", linestyle="--",
                linewidth=1, label="Unit circle")
        ax.plot(sr * np.cos(theta), sr * np.sin(theta), color="red",
                linewidth=1.5, label=f"SR = {sr:.4f}")
        sc = ax.scatter(eigvals.real, eigvals.imag, c=eigvals_abs,
                        cmap="viridis", s=20, zorder=3)
        ax.scatter(eigvals[eigvec_idx].real, eigvals[eigvec_idx].imag,
                   facecolors="none", edgecolors="red", s=80, linewidths=2,
                   zorder=4)
        ax.set_xlabel(r"Re($\lambda$)")
        ax.set_ylabel(r"Im($\lambda$)")
        ax.set_title("Eigenvalue Spectrum")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        plt.colorbar(sc, ax=ax, label=r"$|\lambda|$", shrink=0.8, pad=0.04)
        fig.tight_layout()
        _save(fig, "spectrum")

    # --- Eigenvector heatmaps (dynamic — depends on eigvec_idx) ---
    eigvec_label = (
        f"$\\lambda_{{{eigvec_idx}}}$ = "
        f"{eigvals[eigvec_idx].real:.4f}{eigvals[eigvec_idx].imag:+.4f}j"
    )

    if "magnitude" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, data["mag"][eigvec_idx],
                      f"Magnitude |v| — {eigvec_label}",
                      cbar_label="|v|")
        fig.tight_layout()
        _save(fig, f"magnitude_{eigvec_idx}")

    if "phase" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, data["phase"][eigvec_idx],
                      f"Phase ∠v — {eigvec_label}",
                      cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
                      cbar_label="∠v")
        fig.tight_layout()
        _save(fig, f"phase_{eigvec_idx}")

    if "local_r" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, data["local_kuramoto"][eigvec_idx],
                      f"Local Phase Coherence R — {eigvec_label}",
                      vmin=0, vmax=1, cbar_label="R")
        fig.tight_layout()
        _save(fig, f"local_r_{eigvec_idx}")

    # --- Total activity / total local R (static) ---
    if "total_activity" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, data["activity"],
                      "Total Activity — Σ|λᵢ|·|vᵢ|²",
                      cbar_label="Activity")
        fig.tight_layout()
        _save(fig, "total_activity")

    if "total_local_r" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, data["total_local_kuramoto"],
                      "Local Phase Coherence (weighted total)",
                      vmin=0, vmax=1, cbar_label="R")
        fig.tight_layout()
        _save(fig, "total_local_r")

    # --- Resolvent panels (recomputed from W_res) ---
    if "resolvent_z1" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_z1"],
                      "(I − W)⁻¹ · 1",
                      cmap="RdBu_r", center_zero=True, cbar_label="val")
        fig.tight_layout()
        _save(fig, "resolvent_z1")

    if "resolvent_z_complex_mag" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_z_mag"],
                      "|(e^{i·0.1}I − W)⁻¹ · 1|",
                      cbar_label="|val|")
        fig.tight_layout()
        _save(fig, "resolvent_z_complex_mag")

    if "resolvent_z_complex_phase" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_z_phase"],
                      "∠(e^{i·0.1}I − W)⁻¹ · 1",
                      cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
                      cbar_label="∠")
        fig.tight_layout()
        _save(fig, "resolvent_z_complex_phase")

    if "resolvent_local_r" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_local_r"],
                      "Resolvent Local Phase Coherence",
                      vmin=0, vmax=1, cbar_label="R")
        fig.tight_layout()
        _save(fig, "resolvent_local_r")

    if "resolvent_diag" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_diag"],
                      "diag(I − W)⁻¹",
                      cmap="RdBu_r", center_zero=True, cbar_label="diag")
        fig.tight_layout()
        _save(fig, "resolvent_diag")

    if "relative_self_influence" in panels:
        fig, ax = plt.subplots(figsize=(5, 5))
        _plot_heatmap(ax, res["res_rel_self"],
                      "|R_jj| / Σ_k|R_jk|",
                      vmin=0, vmax=1, cbar_label="rel")
        fig.tight_layout()
        _save(fig, "relative_self_influence")

    # --- Singular value panels ---
    if "sv_spectrum" in panels:
        sv = np.sort(res["sing_vals"])[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(sv)), sv, color="steelblue", width=1.0)
        ax.axhline(sr, color="red", linestyle="--", linewidth=1.5,
                    label=f"SR = {sr:.4f}")
        ax.set_xlabel("Index")
        ax.set_ylabel("σ")
        ax.set_title("Singular Value Spectrum")
        ax.legend()
        fig.tight_layout()
        _save(fig, "sv_spectrum")

    if "sv_distribution" in panels:
        sv = res["sing_vals"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sv, bins=30, color="steelblue", edgecolor="white")
        ax.axvline(sr, color="red", linestyle="--", linewidth=1.5,
                    label=f"SR = {sr:.4f}")
        ax.set_xlabel("σ")
        ax.set_ylabel("Count")
        ax.set_title("Singular Value Distribution")
        ax.legend()
        fig.tight_layout()
        _save(fig, "sv_distribution")

    # --- Phase flow quiver panels ---
    if "resolvent_phase_flow" in panels:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_quiver(ax, res["res_z_phase"].tolist(),
                     "Resolvent Phase Flow", m)
        fig.tight_layout()
        _save(fig, "resolvent_phase_flow")

    if "phase_flow" in panels:
        fig, ax = plt.subplots(figsize=(6, 6))
        _plot_quiver(ax, data["phase"][eigvec_idx],
                     f"Phase Flow — {eigvec_label}", m)
        fig.tight_layout()
        _save(fig, f"phase_flow_{eigvec_idx}")

    print(f"\nExtracted {len(saved)} panels to {output_dir}/")
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = sys.argv[1:]
    html_path = next((a for a in args if not a.startswith("--")), None)
    if html_path is None:
        print("Error: no HTML path provided")
        sys.exit(1)

    def _get_arg(name, default=None):
        prefix = f"--{name}="
        val = next((a[len(prefix):] for a in args if a.startswith(prefix)), None)
        return val if val is not None else default

    eigvec_idx = int(_get_arg("eigvec-idx", "0"))
    output_dir = _get_arg("output-dir")
    fmt = _get_arg("format", "pdf")
    dpi = int(_get_arg("dpi", "300"))
    panels_str = _get_arg("panels")
    panels = panels_str.split(",") if panels_str else None

    extract_panels(
        html_path,
        eigvec_idx=eigvec_idx,
        output_dir=output_dir,
        fmt=fmt,
        panels=panels,
        dpi=dpi,
    )
