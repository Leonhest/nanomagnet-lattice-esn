"""Interactive eigenvector visualization: click eigenvalues to see eigenvectors on the lattice."""

import json
import os
import sys

import numpy as np
from scipy.linalg import schur
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running as ``python -m utils.eigvec_viz`` from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _find_conjugate_pairs(eigvals):
    """Return list where pairs[k] = index of conjugate partner, -1 if real."""
    n = len(eigvals)
    pairs = [-1] * n
    used = set()
    for i in range(n):
        if i in used or np.abs(eigvals[i].imag) < 1e-10:
            continue
        conj = np.conj(eigvals[i])
        best_j = -1
        best_dist = np.inf
        for j in range(n):
            if j == i or j in used:
                continue
            dist = np.abs(eigvals[j] - conj)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0 and best_dist < 1e-8:
            pairs[i] = best_j
            pairs[best_j] = i
            used.add(i)
            used.add(best_j)
    return pairs


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

    # Schur decomposition for eigenvalue removal: W = Q @ T @ Q^H
    # Q is unitary (always invertible), so reconstruction is exact.
    # Removing eigenvalue k: W_modified = W - λ_k * q_k @ q_k^H
    T_schur, Q_schur = schur(W_res, output='complex')
    schur_eigvals = np.diag(T_schur)

    # Match Schur eigenvalues to eigen-ordering (sorted by descending |λ|)
    eig_to_schur = []
    used_schur = set()
    for i in range(n):
        best_j = -1
        best_dist = np.inf
        for j in range(n):
            if j in used_schur:
                continue
            dist = np.abs(eigvals[i] - schur_eigvals[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j
        eig_to_schur.append(best_j)
        used_schur.add(best_j)

    # Conjugate pair mapping: pairs[k] = index of conjugate partner, -1 if real
    conj_pairs = _find_conjugate_pairs(eigvals)

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
    adj_threshold = 1e-8 * np.max(np.abs(W_res))
    adj = ((np.abs(W_res) > adj_threshold) | (np.abs(W_res.T) > adj_threshold))
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

    # --- Resolvent computations -----------------------------------------------
    I = np.eye(n)
    # z=1: (I - W)^{-1}
    res_inv_z1 = np.linalg.inv(I - W_res)
    res_z1 = (res_inv_z1 @ np.ones(n)).reshape(m, m).tolist()

    # z=e^{i*0.1}: complex resolvent
    z2 = np.exp(1j * 0.1)
    res_z_complex = np.linalg.inv(z2 * I - W_res) @ np.ones(n)
    res_z_mag = np.abs(res_z_complex).reshape(m, m).tolist()
    res_z_phase = np.angle(res_z_complex).reshape(m, m).tolist()

    # Local phase coherence of resolvent phase
    res_z_local_r = _local_kuramoto(np.angle(res_z_complex)).reshape(m, m).tolist()

    # diag((I - W)^{-1})  — self-influence per node
    res_diag = np.diag(res_inv_z1).reshape(m, m).tolist()

    # Relative self-influence: |R_jj| / Σ_k |R_jk|  per node
    res_diag_abs = np.abs(np.diag(res_inv_z1))
    res_row_sums = np.sum(np.abs(res_inv_z1), axis=1)
    res_rel_self = (res_diag_abs / res_row_sums).reshape(m, m).tolist()

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
        "Q_re": Q_schur.real.tolist(),
        "Q_im": Q_schur.imag.tolist(),
        "eig_to_schur": eig_to_schur,
        "conj_pairs": conj_pairs,
        "save_dir": os.path.dirname(os.path.abspath(save_path or "eigvec_explorer.html")),
        "W_res_original": W_res.tolist(),
    }

    # --- Build Plotly figure --------------------------------------------------
    # Layout: 4 rows x 4 cols
    #   Row 1: Spectrum [rowspan=2, colspan=2] | Magnitude     | Local R
    #   Row 2:                                 | Phase         |
    #   Row 3: Res(z)@1   | |Res(ze^i)@1|     | angle Res     | Res Local R
    #   Row 4: diag(Res)   | Rel. Self-Infl |              |
    fig = make_subplots(
        rows=4, cols=4,
        column_widths=[0.25, 0.25, 0.25, 0.25],
        row_heights=[0.25, 0.25, 0.25, 0.25],
        specs=[
            [{"rowspan": 2, "colspan": 2}, None, {"type": "heatmap"}, {"type": "heatmap"}],
            [None, None, {"type": "heatmap"}, None],
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}, None, None],
        ],
        subplot_titles=[
            "Eigenvalue Spectrum",
            f"Magnitude |v| \u2014 \u03bb\u2080 = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
            "Local Phase Coherence R",
            f"Phase \u2220v \u2014 \u03bb\u2080 = {eigvals[0].real:.4f}{eigvals[0].imag:+.4f}j",
            "(I \u2212 W)\u207b\u00b9 \u00b7 1",
            "|(e\u2071\u2070\u00b7\u00b9I \u2212 W)\u207b\u00b9 \u00b7 1|",
            "\u2220(e\u2071\u2070\u00b7\u00b9I \u2212 W)\u207b\u00b9 \u00b7 1",
            "Resolvent Local Phase Coherence",
            "diag (I \u2212 W)\u207b\u00b9",
            "|R\u2c7c\u2c7c| / \u03a3\u2096|R\u2c7c\u2096|",
        ],
        horizontal_spacing=0.06,
        vertical_spacing=0.08,
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
                colorbar=dict(title="|\u03bb|", x=-0.08, len=0.30, y=0.77, thickness=12),
                line=dict(width=line_widths, color=line_colors),
            ),
            text=hover_text, hoverinfo="text",
            name="Eigenvalues",
        ),
        row=1, col=1,
    )

    # Trace 3: Magnitude heatmap (row=1, col=3)
    fig.add_trace(
        go.Heatmap(
            z=mag_grids[0], colorscale="Viridis",
            colorbar=dict(title="|v|", x=0.74, len=0.19, y=0.905, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>|v|=%{z:.4f}<extra></extra>",
        ),
        row=1, col=3,
    )

    # Trace 4: Phase heatmap (row=2, col=3)
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
            colorbar=dict(title="\u2220v", x=0.74, len=0.19, y=0.635, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>\u2220v=%{z:.4f}<extra></extra>",
        ),
        row=2, col=3,
    )

    # Trace 5: Local Kuramoto R heatmap (row=1, col=4)
    fig.add_trace(
        go.Heatmap(
            z=local_kuramoto_grids[0], colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="R", x=1.01, len=0.19, y=0.905, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>R=%{z:.4f}<extra></extra>",
        ),
        row=1, col=4,
    )

    # --- Row 3: Resolvent heatmaps (all static) -------------------------------

    # Trace 6: (I - W)^{-1} @ 1  (row=3, col=1)
    fig.add_trace(
        go.Heatmap(
            z=res_z1, colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="val", x=0.21, len=0.19, y=0.365, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>val=%{z:.4f}<extra></extra>",
        ),
        row=3, col=1,
    )

    # Trace 7: |(e^{i*0.5}I - W)^{-1} @ 1|  (row=3, col=2)
    fig.add_trace(
        go.Heatmap(
            z=res_z_mag, colorscale="Viridis",
            colorbar=dict(title="|val|", x=0.47, len=0.19, y=0.365, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>|val|=%{z:.4f}<extra></extra>",
        ),
        row=3, col=2,
    )

    # Trace 8: angle((e^{i*0.5}I - W)^{-1} @ 1)  (row=3, col=3)
    fig.add_trace(
        go.Heatmap(
            z=res_z_phase, colorscale=phase_colorscale,
            zmin=-np.pi, zmax=np.pi,
            colorbar=dict(title="\u2220", x=0.74, len=0.19, y=0.365, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>\u2220=%{z:.4f}<extra></extra>",
        ),
        row=3, col=3,
    )

    # Trace 9: Resolvent local phase coherence (row=3, col=4)
    fig.add_trace(
        go.Heatmap(
            z=res_z_local_r, colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="R", x=1.01, len=0.19, y=0.365, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>R=%{z:.4f}<extra></extra>",
        ),
        row=3, col=4,
    )

    # Trace 10: diag((I - W)^{-1})  (row=4, col=1)
    fig.add_trace(
        go.Heatmap(
            z=res_diag, colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="diag", x=0.21, len=0.19, y=0.095, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>val=%{z:.4f}<extra></extra>",
        ),
        row=4, col=1,
    )

    # Trace 11: Relative self-influence |R_jj| / Σ_k |R_jk|  (row=4, col=2)
    fig.add_trace(
        go.Heatmap(
            z=res_rel_self, colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar=dict(title="rel", x=0.47, len=0.19, y=0.095, thickness=12),
            hovertemplate="row=%{y}, col=%{x}<br>rel=%{z:.4f}<extra></extra>",
        ),
        row=4, col=2,
    )

    # Spectrum axis styling
    fig.update_xaxes(title_text="Re(\u03bb)", row=1, col=1)
    fig.update_yaxes(title_text="Im(\u03bb)", scaleanchor="x", scaleratio=1, constrain="domain", row=1, col=1)

    # Heatmap axes — reversed y for matrix layout
    for r, c in [(1, 3), (1, 4), (2, 3), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2)]:
        fig.update_yaxes(autorange="reversed", row=r, col=c)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=1150,
        margin=dict(l=80, r=30, t=80, b=40),
        showlegend=True,
        legend=dict(x=0.0, y=1.0, yanchor="top", orientation="h"),
    )

    # --- Generate HTML with embedded data + click handler ---------------------
    plot_div = fig.to_html(
        full_html=False, include_plotlyjs="cdn", div_id="eigvec-plot",
        config={"responsive": True},
    )

    # Annotation indices (8 non-None cells):
    #   [0] Eigenvalue Spectrum
    #   [1] Magnitude
    #   [2] Local Phase Coherence R
    #   [3] Phase
    #   [4] Res(z=1)@1
    #   [5] |Res(z=e^i0.5)@1|
    #   [6] angle Res(z=e^i0.5)@1
    #   [7] diag(Res)
    js_handler = """
<script>
(function() {
    var data = JSON.parse(document.getElementById('eigvec-data').textContent);
    var plot = document.getElementById('eigvec-plot');
    var n = data.eigvals_re.length;
    var m = data.m;

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

    // --- Removal mode state ---
    var removalMode = false;
    var removedSet = new Set();

    function eigenLabel(idx) {
        var re = data.eigvals_re[idx].toFixed(4);
        var im = data.eigvals_im[idx] >= 0
            ? '+' + data.eigvals_im[idx].toFixed(4)
            : data.eigvals_im[idx].toFixed(4);
        var abs_val = data.eigvals_abs[idx].toFixed(4);
        return '\\u03bb' + idx + ' = ' + re + im + 'j  |\\u03bb| = ' + abs_val;
    }

    function updateRemovalCount() {
        var el = document.getElementById('removal-count');
        if (removedSet.size > 0) {
            el.textContent = removedSet.size + ' eigenvalue(s) removed';
        } else {
            el.textContent = '';
        }
    }

    function toggleRemoval(idx) {
        if (removedSet.has(idx)) {
            removedSet.delete(idx);
            // Also remove conjugate partner
            var conj = data.conj_pairs[idx];
            if (conj >= 0) removedSet.delete(conj);
        } else {
            removedSet.add(idx);
            // Also add conjugate partner
            var conj = data.conj_pairs[idx];
            if (conj >= 0) removedSet.add(conj);
        }
        updateRemovalVisuals();
        updateRemovalCount();
    }

    function updateRemovalVisuals() {
        // Update marker styles: removed eigenvalues get red X, others normal
        var symbols = new Array(n).fill('circle');
        var sizes = new Array(n).fill(7);
        var widths = new Array(n).fill(0);
        var colors = new Array(n).fill('rgba(0,0,0,0)');

        removedSet.forEach(function(idx) {
            symbols[idx] = 'x';
            sizes[idx] = 10;
        });

        // Keep current selection highlight if not in removal mode
        if (!removalMode && !document.getElementById('activity-btn').classList.contains('active')) {
            widths[currentIdx] = 2.5;
            colors[currentIdx] = 'red';
        }

        Plotly.restyle(plot, {
            'marker.symbol': [symbols],
            'marker.size': [sizes],
            'marker.line.width': [widths],
            'marker.line.color': [colors]
        }, [2]);
    }

    function reconstructMatrix() {
        // Schur-based: W_modified = W_original - sum_{k removed} λ_k * q_k @ q_k^H
        // Q is unitary so reconstruction is exact.
        var nn = n;
        var W = [];
        for (var i = 0; i < nn; i++) {
            W.push(data.W_res_original[i].slice());
        }

        removedSet.forEach(function(eigIdx) {
            var si = data.eig_to_schur[eigIdx];
            var lam_re = data.eigvals_re[eigIdx];
            var lam_im = data.eigvals_im[eigIdx];

            // Schur vector q = Q[:, si]
            // Subtract rank-1: W -= real(λ * q @ q^H)
            // W[i][j] -= real(λ * q[i] * conj(q[j]))
            for (var i = 0; i < nn; i++) {
                var qi_re = data.Q_re[i][si];
                var qi_im = data.Q_im[i][si];
                // λ * q[i]
                var lq_re = lam_re * qi_re - lam_im * qi_im;
                var lq_im = lam_re * qi_im + lam_im * qi_re;

                for (var j = 0; j < nn; j++) {
                    var qj_re = data.Q_re[j][si];
                    var qj_im = data.Q_im[j][si];
                    // real((lq)(conj(qj))) = lq_re*qj_re + lq_im*qj_im
                    W[i][j] -= lq_re * qj_re + lq_im * qj_im;
                }
            }
        });

        return W;
    }

    function buildJSON() {
        var W = reconstructMatrix();

        // Compute new spectral radius (max |eigenvalue| of kept eigenvalues)
        var newSR = 0;
        for (var k = 0; k < n; k++) {
            if (!removedSet.has(k)) {
                var abs_val = data.eigvals_abs[k];
                if (abs_val > newSR) newSR = abs_val;
            }
        }

        var removedIndices = Array.from(removedSet).sort(function(a,b){return a-b;});
        var removedEigvals = removedIndices.map(function(idx) {
            return {
                index: idx,
                real: data.eigvals_re[idx],
                imag: data.eigvals_im[idx],
                abs: data.eigvals_abs[idx]
            };
        });

        return {
            type: "full_matrix",
            size: n,
            grid_shape: [m, m],
            W_res: W,
            metadata: {
                source: "eigenvalue_removal",
                n_eigenvalues_removed: removedSet.size,
                removed_eigenvalue_indices: removedIndices,
                removed_eigenvalues: removedEigvals,
                original_spectral_radius: data.eigvals_abs[0],
                new_spectral_radius: newSR
            }
        };
    }

    async function downloadJSON() {
        var output = buildJSON();
        var jsonStr = JSON.stringify(output);
        var filename = 'W_res_' + removedSet.size + 'removed.json';

        // Try File System Access API to save in the same directory
        if (window.showSaveFilePicker) {
            try {
                var handle = await showSaveFilePicker({
                    suggestedName: filename,
                    types: [{description: 'JSON', accept: {'application/json': ['.json']}}]
                });
                var writable = await handle.createWritable();
                await writable.write(jsonStr);
                await writable.close();
                return;
            } catch (e) {
                if (e.name === 'AbortError') return; // user cancelled
            }
        }

        // Fallback: regular download
        var blob = new Blob([jsonStr], {type: 'application/json'});
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // --- Removal mode toggle ---
    document.getElementById('removal-btn').addEventListener('click', function() {
        removalMode = !removalMode;
        this.classList.toggle('active', removalMode);
        this.textContent = removalMode ? 'Removal Mode ON' : 'Removal Mode';
        updateRemovalVisuals();
    });

    // --- Download button ---
    document.getElementById('download-btn').addEventListener('click', downloadJSON);

    // Click eigenvalue -> show that eigenvector OR toggle removal
    plot.on('plotly_click', function(eventData) {
        if (eventData.points[0].curveNumber !== 2) return;
        var idx = eventData.points[0].pointIndex;

        if (removalMode) {
            toggleRemoval(idx);
            return;
        }

        currentIdx = idx;

        // Update magnitude (trace 3), phase (trace 4), local R (trace 5)
        Plotly.restyle(plot, {z: [data.mag[idx]]}, [3]);
        Plotly.restyle(plot, {z: [data.phase[idx]], colorscale: [phaseCS], zmin: [-Math.PI], zmax: [Math.PI]}, [4]);
        Plotly.restyle(plot, {z: [data.local_kuramoto[idx]]}, [5]);

        // Highlight selected eigenvalue (preserve removal visuals)
        var widths = new Array(n).fill(0);
        var colors = new Array(n).fill('rgba(0,0,0,0)');
        widths[idx] = 2.5;
        colors[idx] = 'red';
        var symbols = new Array(n).fill('circle');
        var sizes = new Array(n).fill(7);
        removedSet.forEach(function(ri) {
            symbols[ri] = 'x';
            sizes[ri] = 10;
        });
        Plotly.restyle(plot, {
            'marker.line.width': [widths],
            'marker.line.color': [colors],
            'marker.symbol': [symbols],
            'marker.size': [sizes]
        }, [2]);

        // Update subplot titles
        var label = eigenLabel(idx);
        var re = data.eigvals_re[idx].toFixed(4);
        var im = data.eigvals_im[idx] >= 0
            ? '+' + data.eigvals_im[idx].toFixed(4)
            : data.eigvals_im[idx].toFixed(4);
        var shortLabel = '\\u03bb' + idx + ' = ' + re + im + 'j';

        var annotations = plot.layout.annotations || [];
        if (annotations.length >= 10) {
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
            Plotly.restyle(plot, {z: [data.activity]}, [3]);
            var blank = [];
            for (var r = 0; r < m; r++) {
                var row = [];
                for (var c = 0; c < m; c++) row.push(0);
                blank.push(row);
            }
            Plotly.restyle(plot, {z: [blank], colorscale: [[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']]], zmin: [0], zmax: [1]}, [4]);
            Plotly.restyle(plot, {z: [data.total_local_kuramoto]}, [5]);

            var widths = new Array(n).fill(0);
            var colors = new Array(n).fill('rgba(0,0,0,0)');
            var symbols = new Array(n).fill('circle');
            var sizes = new Array(n).fill(7);
            removedSet.forEach(function(ri) {
                symbols[ri] = 'x';
                sizes[ri] = 10;
            });
            Plotly.restyle(plot, {
                'marker.line.width': [widths],
                'marker.line.color': [colors],
                'marker.symbol': [symbols],
                'marker.size': [sizes]
            }, [2]);

            var annotations = plot.layout.annotations || [];
            if (annotations.length >= 10) {
                annotations[1].text = 'Total Activity \\u2014 \\u03a3 |\\u03bb\\u1d62|\\u00b7|v\\u1d62|\\u00b2';
                annotations[2].text = 'Local Phase Coherence (weighted total)';
                annotations[3].text = '(inactive during Total Activity)';
                Plotly.relayout(plot, {annotations: annotations});
            }
        } else {
            var idx = currentIdx;
            Plotly.restyle(plot, {z: [data.mag[idx]]}, [3]);
            Plotly.restyle(plot, {z: [data.phase[idx]], colorscale: [phaseCS], zmin: [-Math.PI], zmax: [Math.PI]}, [4]);
            Plotly.restyle(plot, {z: [data.local_kuramoto[idx]]}, [5]);

            var widths = new Array(n).fill(0);
            var colors = new Array(n).fill('rgba(0,0,0,0)');
            widths[idx] = 2.5;
            colors[idx] = 'red';
            var symbols = new Array(n).fill('circle');
            var sizes = new Array(n).fill(7);
            removedSet.forEach(function(ri) {
                symbols[ri] = 'x';
                sizes[ri] = 10;
            });
            Plotly.restyle(plot, {
                'marker.line.width': [widths],
                'marker.line.color': [colors],
                'marker.symbol': [symbols],
                'marker.size': [sizes]
            }, [2]);

            var re = data.eigvals_re[idx].toFixed(4);
            var im = data.eigvals_im[idx] >= 0
                ? '+' + data.eigvals_im[idx].toFixed(4)
                : data.eigvals_im[idx].toFixed(4);
            var label = eigenLabel(idx);
            var shortLabel = '\\u03bb' + idx + ' = ' + re + im + 'j';

            var annotations = plot.layout.annotations || [];
            if (annotations.length >= 10) {
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
  #eigvec-plot {{ width: 100%; min-height: 1100px; }}
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
  #removal-btn {{
    padding: 8px 18px; margin-left: 16px; cursor: pointer;
    border: 2px solid #b33; border-radius: 6px;
    background: #f5f5f5; font-size: 14px; font-weight: 600;
    color: #b33; transition: all 0.15s;
  }}
  #removal-btn:hover {{ background: #fdd; }}
  #removal-btn.active {{ background: #b33; color: #fff; }}
  #download-btn {{
    padding: 8px 18px; margin-left: 8px; cursor: pointer;
    border: 2px solid #36a; border-radius: 6px;
    background: #f5f5f5; font-size: 14px; font-weight: 600;
    color: #36a; transition: all 0.15s;
  }}
  #download-btn:hover {{ background: #def; }}
  #download-btn:disabled {{ opacity: 0.4; cursor: default; }}
  #removal-count {{
    margin-left: 10px; font-size: 14px; color: #b33; font-weight: 600;
  }}
</style>
</head>
<body>
<div id="info">
  Click an eigenvalue on the spectrum to view its eigenvector as magnitude, phase, and local phase coherence heatmaps on the {m}&times;{m} lattice.
  <button id="activity-btn">Total Activity</button>
  <button id="kuramoto-btn">Color: |&lambda;|</button>
  <button id="removal-btn">Removal Mode</button>
  <button id="download-btn">Download W_res JSON</button>
  <span id="removal-count"></span>
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
