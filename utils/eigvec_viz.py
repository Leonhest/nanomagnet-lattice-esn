"""Interactive eigenvector visualization: click eigenvalues to see eigenvectors on the lattice."""

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running as ``python -m utils.eigvec_viz`` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def eigenvector_viz(W_res, *, target_sr=None, save_path=None, title="Eigenvector Explorer"):
    """Interactive eigenvalue/eigenvector explorer -> standalone HTML.

    Parameters
    ----------
    W_res : np.ndarray
        Square reservoir weight matrix.
    target_sr : float or None
        Target spectral radius.  When provided, W_res is rescaled to this SR
        before analysis so that eigenvalues/resolvent match the actual ESN.
    save_path : str or None
        Where to save the HTML. Defaults to ``eigvec_explorer.html`` in cwd.
    title : str
        Page title.

    Returns
    -------
    str
        Path to the saved HTML file.
    """
    # --- Spectral radius scaling ---------------------------------------------
    if target_sr is not None:
        actual_sr = np.max(np.abs(np.linalg.eigvals(W_res)))
        if actual_sr > 0:
            W_res = W_res * (target_sr / actual_sr)

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

    # Total activity: sum_i |lambda_i| * |v_i|^2  per node
    abs_eigvals = np.abs(eigvals)
    activity = np.zeros(n)
    for i in range(n):
        activity += abs_eigvals[i] * np.abs(eigvecs[:, i]) ** 2
    activity_grid = activity.reshape(m, m).tolist()

    # Kuramoto order parameter per eigenvector: R_k = |mean(exp(i*angle(v_k)))|
    # R_k ~ 1: all nodes in sync (breathing mode), R_k ~ 0: spread phases (wave)
    kuramoto = np.array([
        float(np.abs(np.mean(np.exp(1j * np.angle(eigvecs[:, i])))))
        for i in range(n)
    ])

    # Local Kuramoto per node: R_j = |mean(exp(i*angle(v[j and neighbors])))|
    # Measures local phase coherence of node WITH its neighborhood.
    # Adjacency mask for neighbors (symmetric union of in/out edges, no self-loops)
    adj = ((W_res != 0) | (W_res.T != 0))
    np.fill_diagonal(adj, False)
    # Degree per node (+1 for the node itself)
    deg_plus1 = adj.sum(axis=1).astype(float) + 1.0

    def _local_kuramoto(phases):
        """Vectorized local Kuramoto for all nodes given phase vector."""
        phasors = np.exp(1j * phases)          # (n,)
        nbr_sum = adj @ phasors + phasors      # (n,) node + neighbors
        return np.abs(nbr_sum) / deg_plus1

    # Per-eigenvector local Kuramoto grids + weighted total
    local_kuramoto_grids = []
    total_local_kuramoto = np.zeros(n)
    for i in range(n):
        lk = _local_kuramoto(np.angle(eigvecs[:, i]))
        local_kuramoto_grids.append(lk.reshape(m, m).tolist())
        total_local_kuramoto += abs_eigvals[i] * lk
    total_local_kuramoto /= abs_eigvals.sum()
    total_local_kuramoto_grid = total_local_kuramoto.reshape(m, m).tolist()

    # Resolvent: (I - W)^{-1}
    resolvent = np.linalg.inv(np.eye(n) - W_res)
    resolvent_grid = resolvent.tolist()

    eigvec_data = {
        "mag": mag_grids,
        "phase": phase_grids,
        "local_kuramoto": local_kuramoto_grids,
        "eigvals_re": eigvals.real.tolist(),
        "eigvals_im": eigvals.imag.tolist(),
        "eigvals_abs": np.abs(eigvals).tolist(),
        "kuramoto": kuramoto.tolist(),
        "m": m,
        "activity": activity_grid,
        "total_local_kuramoto": total_local_kuramoto_grid,
        "resolvent": resolvent_grid,
    }

    # --- Build Plotly figure --------------------------------------------------
    fig = make_subplots(
        rows=2, cols=3,
        column_widths=[0.40, 0.30, 0.30],
        row_heights=[0.5, 0.5],
        specs=[
            [{"rowspan": 2}, {"type": "heatmap"}, {"type": "heatmap"}],
            [None, {"type": "heatmap"}, {"type": "heatmap"}],
        ],
        subplot_titles=[
            "Eigenvalue Spectrum",
            f"Magnitude |v| \u2014 \u03bb\u2080 = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
            "Local Phase Coherence R",
            f"Phase \u2220v \u2014 \u03bb\u2080 = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
            "Resolvent (I \u2212 W)\u207b\u00b9",
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
        f"\u03bb = {eigvals[i].real:.4f}{eigvals[i].imag:+.4f}j<br>|\u03bb| = {np.abs(eigvals[i]):.4f}<br>R = {kuramoto[i]:.3f}<br>index = {i}"
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
                colorbar=dict(title="|\u03bb|", x=-0.07, len=0.45, y=0.5, thickness=12),
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
            colorbar=dict(title="|v|", x=0.69, len=0.35, y=0.78, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>|v|=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    # Trace 4: Phase heatmap (cyclic colorscale so -pi and pi are the same color)
    phase_colorscale = [
        [0.0,  "rgb(225,216,226)"],
        [0.1,  "rgb(165,191,202)"],
        [0.2,  "rgb(109,144,191)"],
        [0.3,  "rgb(94,87,176)"],
        [0.4,  "rgb(83,29,124)"],
        [0.5,  "rgb(47,20,54)"],
        [0.6,  "rgb(99,24,75)"],
        [0.7,  "rgb(158,59,79)"],
        [0.8,  "rgb(192,116,93)"],
        [0.9,  "rgb(208,178,158)"],
        [1.0,  "rgb(225,216,225)"],
    ]
    fig.add_trace(
        go.Heatmap(
            z=phase_grids[0], colorscale=phase_colorscale,
            zmin=-np.pi, zmax=np.pi,
            colorbar=dict(title="\u2220v", x=0.69, len=0.35, y=0.22, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>\u2220v=%{z:.4f}<extra></extra>",
        ),
        row=2, col=2,
    )

    # Trace 5: Local Kuramoto R heatmap (default: dominant eigenvector)
    fig.add_trace(
        go.Heatmap(
            z=local_kuramoto_grids[0], colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="R", x=1.01, len=0.35, y=0.78, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>R=%{z:.4f}<extra></extra>",
        ),
        row=1, col=3,
    )

    # Trace 6: Resolvent heatmap (static, full n x n matrix)
    res_abs_max = np.max(np.abs(resolvent))
    fig.add_trace(
        go.Heatmap(
            z=resolvent_grid, colorscale="RdBu_r",
            zmid=0, zmin=-res_abs_max, zmax=res_abs_max,
            colorbar=dict(title="(I\u2212W)\u207b\u00b9", x=1.01, len=0.35, y=0.22, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>val=%{z:.4f}<extra></extra>",
        ),
        row=2, col=3,
    )

    # Spectrum axis styling
    fig.update_xaxes(title_text="Re(\u03bb)", row=1, col=1)
    fig.update_yaxes(title_text="Im(\u03bb)", scaleanchor="x", scaleratio=1, constrain="domain", row=1, col=1)

    # Heatmap axes — reversed y for matrix layout
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=3)
    fig.update_yaxes(autorange="reversed", row=2, col=3)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=750,
        margin=dict(l=80, r=30, t=80, b=40),
        showlegend=True,
        legend=dict(x=0.0, y=0.98, orientation="h"),
    )

    # --- Generate HTML with embedded data + click handler ---------------------
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="eigvec-plot",
        config={"responsive": True},
    )

    js_handler = """
<script>
(function() {
    var data = JSON.parse(document.getElementById('eigvec-data').textContent);
    var plot = document.getElementById('eigvec-plot');
    var n = data.eigvals_re.length;

    var phaseCS = [
        [0.0,  'rgb(225,216,226)'],
        [0.1,  'rgb(165,191,202)'],
        [0.2,  'rgb(109,144,191)'],
        [0.3,  'rgb(94,87,176)'],
        [0.4,  'rgb(83,29,124)'],
        [0.5,  'rgb(47,20,54)'],
        [0.6,  'rgb(99,24,75)'],
        [0.7,  'rgb(158,59,79)'],
        [0.8,  'rgb(192,116,93)'],
        [0.9,  'rgb(208,178,158)'],
        [1.0,  'rgb(225,216,225)']
    ];

    var currentIdx = 0;

    function eigenLabel(idx) {
        var re = data.eigvals_re[idx].toFixed(4);
        var im = data.eigvals_im[idx] >= 0
            ? '+' + data.eigvals_im[idx].toFixed(4)
            : data.eigvals_im[idx].toFixed(4);
        var abs_val = data.eigvals_abs[idx].toFixed(4);
        return '\\u03bb' + idx + ' = ' + re + im + 'j  |\\u03bb| = ' + abs_val;
    }

    // Click eigenvalue -> show that eigenvector
    plot.on('plotly_click', function(eventData) {
        if (eventData.points[0].curveNumber !== 2) return;
        var idx = eventData.points[0].pointIndex;
        currentIdx = idx;

        // Update magnitude (trace 3), phase (trace 4), local R (trace 5)
        Plotly.restyle(plot, {z: [data.mag[idx]]}, [3]);
        Plotly.restyle(plot, {z: [data.phase[idx]], colorscale: [phaseCS], zmin: [-Math.PI], zmax: [Math.PI]}, [4]);
        Plotly.restyle(plot, {z: [data.local_kuramoto[idx]]}, [5]);

        // Highlight selected eigenvalue
        var widths = new Array(n).fill(0);
        var colors = new Array(n).fill('rgba(0,0,0,0)');
        widths[idx] = 2.5;
        colors[idx] = 'red';
        Plotly.restyle(plot, {'marker.line.width': [widths], 'marker.line.color': [colors]}, [2]);

        // Update subplot titles
        var label = eigenLabel(idx);
        var re = data.eigvals_re[idx].toFixed(4);
        var im = data.eigvals_im[idx] >= 0
            ? '+' + data.eigvals_im[idx].toFixed(4)
            : data.eigvals_im[idx].toFixed(4);
        var shortLabel = '\\u03bb' + idx + ' = ' + re + im + 'j';

        var annotations = plot.layout.annotations || [];
        if (annotations.length >= 5) {
            annotations[1].text = 'Magnitude |v| \\u2014 ' + label;
            annotations[2].text = 'Local Phase Coherence \\u2014 ' + shortLabel;
            annotations[3].text = 'Phase \\u2220v \\u2014 ' + shortLabel;
            Plotly.relayout(plot, {annotations: annotations});
        }

        document.getElementById('activity-btn').classList.remove('active');
    });

    // Total activity button
    document.getElementById('activity-btn').addEventListener('click', function() {
        this.classList.toggle('active');
        var isActive = this.classList.contains('active');

        if (isActive) {
            // Magnitude -> total activity
            Plotly.restyle(plot, {z: [data.activity]}, [3]);
            // Phase -> blank (not meaningful)
            var m = data.m;
            var blank = [];
            for (var r = 0; r < m; r++) {
                var row = [];
                for (var c = 0; c < m; c++) row.push(0);
                blank.push(row);
            }
            Plotly.restyle(plot, {z: [blank], colorscale: [[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']]], zmin: [0], zmax: [1]}, [4]);
            // Local R -> weighted total
            Plotly.restyle(plot, {z: [data.total_local_kuramoto]}, [5]);

            // Clear eigenvalue highlights
            var widths = new Array(n).fill(0);
            var colors = new Array(n).fill('rgba(0,0,0,0)');
            Plotly.restyle(plot, {'marker.line.width': [widths], 'marker.line.color': [colors]}, [2]);

            var annotations = plot.layout.annotations || [];
            if (annotations.length >= 5) {
                annotations[1].text = 'Total Activity \\u2014 \\u03a3 |\\u03bb\\u1d62|\\u00b7|v\\u1d62|\\u00b2';
                annotations[2].text = 'Local Phase Coherence (weighted total)';
                annotations[3].text = '(inactive during Total Activity)';
                Plotly.relayout(plot, {annotations: annotations});
            }
        } else {
            // Revert to current eigenvector
            var idx = currentIdx;
            Plotly.restyle(plot, {z: [data.mag[idx]]}, [3]);
            Plotly.restyle(plot, {z: [data.phase[idx]], colorscale: [phaseCS], zmin: [-Math.PI], zmax: [Math.PI]}, [4]);
            Plotly.restyle(plot, {z: [data.local_kuramoto[idx]]}, [5]);

            var widths = new Array(n).fill(0);
            var colors = new Array(n).fill('rgba(0,0,0,0)');
            widths[idx] = 2.5;
            colors[idx] = 'red';
            Plotly.restyle(plot, {'marker.line.width': [widths], 'marker.line.color': [colors]}, [2]);

            var re = data.eigvals_re[idx].toFixed(4);
            var im = data.eigvals_im[idx] >= 0
                ? '+' + data.eigvals_im[idx].toFixed(4)
                : data.eigvals_im[idx].toFixed(4);
            var label = eigenLabel(idx);
            var shortLabel = '\\u03bb' + idx + ' = ' + re + im + 'j';

            var annotations = plot.layout.annotations || [];
            if (annotations.length >= 5) {
                annotations[1].text = 'Magnitude |v| \\u2014 ' + label;
                annotations[2].text = 'Local Phase Coherence \\u2014 ' + shortLabel;
                annotations[3].text = 'Phase \\u2220v \\u2014 ' + shortLabel;
                Plotly.relayout(plot, {annotations: annotations});
            }
        }
    });

    // Kuramoto color toggle
    var showKuramoto = false;
    document.getElementById('kuramoto-btn').addEventListener('click', function() {
        showKuramoto = !showKuramoto;
        if (showKuramoto) {
            Plotly.restyle(plot, {
                'marker.color': [data.kuramoto],
                'marker.colorbar.title': 'R',
                'marker.cmin': 0,
                'marker.cmax': 1
            }, [2]);
            this.textContent = 'Color: R';
        } else {
            Plotly.restyle(plot, {
                'marker.color': [data.eigvals_abs],
                'marker.colorbar.title': '|\\u03bb|',
                'marker.cmin': null,
                'marker.cmax': null
            }, [2]);
            this.textContent = 'Color: |\\u03bb|';
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
  #eigvec-plot {{ width: 100%; min-height: 750px; }}
  #info {{ color: #666; margin-bottom: 10px; font-size: 14px; }}
  #activity-btn {{
    padding: 8px 18px; margin-left: 16px; cursor: pointer;
    border: 2px solid #4a4a4a; border-radius: 6px;
    background: #f5f5f5; font-size: 14px; font-weight: 600;
    transition: all 0.15s;
  }}
  #activity-btn:hover, #kuramoto-btn:hover {{ background: #e0e0e0; }}
  #activity-btn.active {{ background: #4a4a4a; color: #fff; }}
  #kuramoto-btn {{
    padding: 8px 18px; margin-left: 8px; cursor: pointer;
    border: 2px solid #4a4a4a; border-radius: 6px;
    background: #f5f5f5; font-size: 14px; font-weight: 600;
    transition: all 0.15s;
  }}
</style>
</head>
<body>
<div id="info">
  Click an eigenvalue on the spectrum to view its eigenvector as magnitude, phase, and local phase coherence heatmaps on the {m}&times;{m} lattice.
  <button id="activity-btn">Total Activity</button>
  <button id="kuramoto-btn">Color: |&lambda;|</button>
</div>
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


def eigenvector_viz_from_tile(tile_path, *, lattice_size=400, target_sr=None,
                              save_path=None):
    """Convenience: tile JSON -> W_res -> eigenvector_viz."""
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

    title = f"Eigenvector Explorer \u2014 {os.path.basename(tile_path)} ({lattice_size} nodes)"
    return eigenvector_viz(W_res, target_sr=target_sr, save_path=save_path,
                           title=title)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m utils.eigvec_viz <tile.json> [--size=N] [--sr=X]")
        sys.exit(1)

    args = sys.argv[1:]
    json_path = next((a for a in args if a.endswith(".json")), None)
    if json_path is None:
        print("Error: no .json tile path provided")
        sys.exit(1)

    size_arg = next((a for a in args if a.startswith("--size=")), None)
    lattice_size = int(size_arg.split("=")[1]) if size_arg else 400

    sr_arg = next((a for a in args if a.startswith("--sr=")), None)
    target_sr = float(sr_arg.split("=")[1]) if sr_arg else None

    eigenvector_viz_from_tile(json_path, lattice_size=lattice_size,
                              target_sr=target_sr)
