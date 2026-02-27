"""Interactive eigenvector visualization: click eigenvalues to see eigenvectors on the lattice."""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running as ``python -m utils.eigvec_viz`` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def eigenvector_viz(W_res, *, save_path=None, title="Eigenvector Explorer"):
    """Interactive eigenvalue/eigenvector explorer → standalone HTML.

    Parameters
    ----------
    W_res : np.ndarray
        Square reservoir weight matrix.
    save_path : str or None
        Where to save the HTML. Defaults to ``eigvec_explorer.html`` in cwd.
    title : str
        Page title.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    n = W_res.shape[0]
    m = int(np.sqrt(n))
    assert m * m == n, f"Matrix size {n} is not a perfect square"

    # Eigen-decomposition, sorted by descending |lambda|
    eigvals, eigvecs = np.linalg.eig(W_res)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sr = np.max(np.abs(eigvals))

    # Pre-compute magnitude and phase grids for every eigenvector
    mag_grids = []
    phase_grids = []
    for i in range(n):
        v = eigvecs[:, i]
        mag_grids.append(np.abs(v).reshape(m, m).tolist())
        phase_grids.append(np.angle(v).reshape(m, m).tolist())

    eigvec_data = {
        "mag": mag_grids,
        "phase": phase_grids,
        "eigvals_re": eigvals.real.tolist(),
        "eigvals_im": eigvals.imag.tolist(),
        "eigvals_abs": np.abs(eigvals).tolist(),
        "m": m,
    }

    # --- Build Plotly figure --------------------------------------------------
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.55, 0.45],
        row_heights=[0.5, 0.5],
        specs=[
            [{"rowspan": 2}, {"type": "heatmap"}],
            [None, {"type": "heatmap"}],
        ],
        subplot_titles=[
            "Eigenvalue Spectrum",
            f"Magnitude |v| — λ₀ = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
            "",
            f"Phase ∠v — λ₀ = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    # Trace 0: Unit circle
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(
        go.Scatter(
            x=np.cos(theta).tolist(), y=np.sin(theta).tolist(),
            mode="lines", line=dict(color="gray", dash="dash", width=1),
            name="Unit circle", hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Trace 1: Spectral radius circle
    fig.add_trace(
        go.Scatter(
            x=(sr * np.cos(theta)).tolist(), y=(sr * np.sin(theta)).tolist(),
            mode="lines", line=dict(color="red", width=1.5),
            name=f"SR = {sr:.4f}", hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # Trace 2: Eigenvalue scatter
    hover_text = [
        f"λ = {eigvals[i].real:.4f}{eigvals[i].imag:+.4f}j<br>|λ| = {np.abs(eigvals[i]):.4f}<br>index = {i}"
        for i in range(n)
    ]
    # Marker line: highlight the dominant eigenvalue by default
    line_widths = [2.5] + [0] * (n - 1)
    line_colors = ["red"] + ["rgba(0,0,0,0)"] * (n - 1)
    fig.add_trace(
        go.Scatter(
            x=eigvals.real.tolist(), y=eigvals.imag.tolist(),
            mode="markers",
            marker=dict(
                size=7,
                color=np.abs(eigvals).tolist(),
                colorscale="Viridis",
                colorbar=dict(title="|λ|", x=-0.05, len=0.45, y=0.5),
                line=dict(width=line_widths, color=line_colors),
            ),
            text=hover_text, hoverinfo="text",
            name="Eigenvalues",
        ),
        row=1, col=1,
    )

    # Trace 3: Magnitude heatmap (default: dominant eigenvector)
    fig.add_trace(
        go.Heatmap(
            z=mag_grids[0], colorscale="Viridis",
            colorbar=dict(title="|v|", x=1.02, len=0.4, y=0.78),
            hovertemplate="row=%{y}, col=%{x}<br>|v|=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # Trace 4: Phase heatmap
    fig.add_trace(
        go.Heatmap(
            z=phase_grids[0], colorscale="RdBu", zmid=0, zmin=-np.pi, zmax=np.pi,
            colorbar=dict(title="∠v", x=1.02, len=0.4, y=0.22),
            hovertemplate="row=%{y}, col=%{x}<br>∠v=%{z:.4f}<extra></extra>",
        ),
        row=2, col=2,
    )

    # Spectrum axis styling
    fig.update_xaxes(title_text="Re(λ)", row=1, col=1)
    fig.update_yaxes(title_text="Im(λ)", scaleanchor="x", scaleratio=1, row=1, col=1)

    # Heatmap axes
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=750, width=1200,
        showlegend=True,
        legend=dict(x=0.0, y=1.02, orientation="h"),
    )

    # --- Generate HTML with embedded data + click handler ---------------------
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="eigvec-plot",
    )

    js_handler = """
<script>
(function() {
    var data = JSON.parse(document.getElementById('eigvec-data').textContent);
    var plot = document.getElementById('eigvec-plot');
    var n = data.eigvals_re.length;

    plot.on('plotly_click', function(eventData) {
        // Only respond to clicks on trace 2 (eigenvalue scatter)
        if (eventData.points[0].curveNumber !== 2) return;
        var idx = eventData.points[0].pointIndex;

        // Update magnitude heatmap (trace 3)
        Plotly.restyle(plot, {z: [data.mag[idx]]}, [3]);
        // Update phase heatmap (trace 4)
        Plotly.restyle(plot, {z: [data.phase[idx]]}, [4]);

        // Highlight selected point with red ring, clear others
        var widths = new Array(n).fill(0);
        var colors = new Array(n).fill('rgba(0,0,0,0)');
        widths[idx] = 2.5;
        colors[idx] = 'red';
        Plotly.restyle(plot, {'marker.line.width': [widths], 'marker.line.color': [colors]}, [2]);

        // Update subplot titles
        var re = data.eigvals_re[idx].toFixed(4);
        var im = data.eigvals_im[idx] >= 0
            ? '+' + data.eigvals_im[idx].toFixed(4)
            : data.eigvals_im[idx].toFixed(4);
        var abs_val = data.eigvals_abs[idx].toFixed(4);
        var label = 'λ' + idx + ' = ' + re + im + 'j  |λ| = ' + abs_val;

        var layout = plot.layout;
        var annotations = layout.annotations || [];
        // annotations[1] = magnitude title, annotations[3] = phase title
        if (annotations.length >= 4) {
            annotations[1].text = 'Magnitude |v| — ' + label;
            annotations[3].text = 'Phase ∠v — ' + label;
            Plotly.relayout(plot, {annotations: annotations});
        }
    });
})();
</script>
"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 20px; font-family: sans-serif; }}
  #info {{ color: #666; margin-bottom: 10px; font-size: 14px; }}
</style>
</head>
<body>
<div id="info">Click an eigenvalue on the spectrum to view its eigenvector as magnitude and phase heatmaps on the {m}&times;{m} lattice.</div>
<script type="application/json" id="eigvec-data">
{json.dumps(eigvec_data)}
</script>
{plot_div}
{js_handler}
</body>
</html>
"""

    if save_path is None:
        save_path = "eigvec_explorer.html"
    with open(save_path, "w") as f:
        f.write(html)
    print(f"Saved eigenvector explorer to {save_path}")
    return save_path


def eigenvector_viz_from_tile(tile_path, *, lattice_size=400, save_path=None):
    """Convenience: tile JSON → W_res → eigenvector_viz."""
    import networkx as nx
    from matrix import load_tile_with_metadata, tile_to_lattice

    tile_G, metadata = load_tile_with_metadata(tile_path)
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
        save_path = f"{stem}_eigvec.html"

    title = f"Eigenvector Explorer — {os.path.basename(tile_path)} ({lattice_size} nodes)"
    return eigenvector_viz(W_res, save_path=save_path, title=title)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.eigvec_viz <tile.json> [--size=N]")
        sys.exit(1)

    args = sys.argv[1:]
    json_path = next((a for a in args if a.endswith(".json")), None)
    if json_path is None:
        print("Error: no .json tile path provided")
        sys.exit(1)

    size_arg = next((a for a in args if a.startswith("--size=")), None)
    lattice_size = int(size_arg.split("=")[1]) if size_arg else 400

    eigenvector_viz_from_tile(json_path, lattice_size=lattice_size)
