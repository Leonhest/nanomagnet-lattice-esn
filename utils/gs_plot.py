import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def _display_label(val):
    """Shorten file paths to their filename (without extension) for plot labels."""
    if isinstance(val, str) and '/' in val and '.' in os.path.basename(val):
        name = os.path.splitext(os.path.basename(val))[0]
        if name.startswith("best_"):
            name = name[len("best_"):]
        return name
    return val


def _encode_column(values):
    """
    Encode a flat list of values into floats for plotting.

    Returns:
        (np.ndarray, tick_labels_or_None)
        If all values are numeric (or numeric + None), tick_labels is None
        and the column is treated as numeric (None maps to a sentinel value
        below the minimum so it doesn't get prioritised as categorical).
        Otherwise, tick_labels is a list of display strings for each integer index.
    """
    has_none = any(v is None for v in values)
    non_none = [v for v in values if v is not None]

    # If non-None values are all numeric, keep column numeric
    if has_none and non_none:
        try:
            numeric = np.array(non_none, dtype=float)
            # Place None below the numeric range so it sorts first on the axis
            sentinel = float(np.min(numeric)) - 1.0
            encoded = np.array([sentinel if v is None else float(v) for v in values])
            return encoded, None
        except (ValueError, TypeError):
            pass

    try:
        encoded = np.array(values, dtype=float)
        return encoded, None
    except (ValueError, TypeError):
        unique_vals = sorted(set(values), key=lambda v: str(v))
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        encoded = np.array([val_to_idx[v] for v in values], dtype=float)
        tick_labels = [_display_label(v) for v in unique_vals]
        return encoded, tick_labels


def _plot_1d(x, y, tick_labels, x_label, y_label, title, filepath):
    """Plot a 1D line chart and save as PNG."""
    order = np.argsort(x)
    x, y = x[order], y[order]

    plt.figure()
    plt.plot(x, y, marker="o")
    if tick_labels is not None:
        plt.xticks(range(len(tick_labels)), tick_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close()


def _select_line_group(results):
    """
    Decide which of 2 params should be the line group vs x-axis.

    Categorical params (those with tick_labels from _encode_column) become line
    groups over numeric params. If both are the same type, the param with fewer
    unique values becomes the line group.

    Returns:
        (group_idx, x_idx, unique_counts)
    """
    raw = [[list(pv)[i] for pv, _ in results] for i in range(2)]
    unique_counts = [len(set(col)) for col in raw]
    _, ticks0 = _encode_column(raw[0])
    _, ticks1 = _encode_column(raw[1])
    is_cat = [ticks0 is not None, ticks1 is not None]

    if is_cat[0] and not is_cat[1]:
        group_idx, x_idx = 0, 1
    elif is_cat[1] and not is_cat[0]:
        group_idx, x_idx = 1, 0
    elif unique_counts[0] <= unique_counts[1]:
        group_idx, x_idx = 0, 1
    else:
        group_idx, x_idx = 1, 0

    return group_idx, x_idx, unique_counts


def _plot_2d_heatmap(param_names, results, exp_path, metric_label, suffix):
    """
    Plot a 2D heatmap: param_names[0] on x-axis, param_names[1] on y-axis,
    metric as cell color with text annotations. Saved as PNG.
    """
    display_names = [n.split(".")[-1] for n in param_names]

    raw_x = [list(pv)[0] for pv, _ in results]
    raw_y = [list(pv)[1] for pv, _ in results]
    scores = np.array([s for _, s in results], dtype=float)

    x_enc, x_ticks = _encode_column(raw_x)
    y_enc, y_ticks = _encode_column(raw_y)

    x_unique = np.unique(x_enc)
    y_unique = np.unique(y_enc)

    grid = np.full((len(y_unique), len(x_unique)), np.nan)
    x_idx_map = {v: i for i, v in enumerate(x_unique)}
    y_idx_map = {v: i for i, v in enumerate(y_unique)}

    for xi, yi, s in zip(x_enc, y_enc, scores):
        grid[y_idx_map[yi], x_idx_map[xi]] = s

    fig, ax = plt.subplots(figsize=(max(6, len(x_unique) * 0.8 + 2), max(5, len(y_unique) * 0.6 + 2)))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="plasma")
    fig.colorbar(im, ax=ax, label=metric_label)

    # Annotate cells
    for j in range(len(y_unique)):
        for i in range(len(x_unique)):
            val = grid[j, i]
            if np.isfinite(val):
                ax.text(i, j, f"{val:.4f}", ha="center", va="center", fontsize=7,
                        color="white" if val < (np.nanmin(grid) + np.nanmax(grid)) / 2 else "black")

    x_labels = x_ticks if x_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in x_unique]
    y_labels = y_ticks if y_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in y_unique]

    ax.set_xticks(range(len(x_unique)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(y_unique)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel(display_names[0])
    ax.set_ylabel(display_names[1])
    ax.set_title(f"{metric_label} Heatmap")
    plt.tight_layout()

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}{suffix}_heatmap.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


def _plot_2d_lines(param_names, results, exp_path, metric_label, suffix, *, max_groups=6):
    """
    Plot overlaid lines. Uses _select_line_group to pick the line group
    (categorical preferred) vs x-axis. Skips if line group has more than
    max_groups unique values. Saved as PNG.
    """
    display_names = [n.split(".")[-1] for n in param_names]

    group_idx, x_idx, unique_counts = _select_line_group(results)
    if unique_counts[group_idx] > max_groups:
        print(f"Skipping line plot for {' vs '.join(display_names)}: "
              f"{display_names[group_idx]} has {unique_counts[group_idx]} groups (max {max_groups})")
        return

    groups = defaultdict(list)
    for pv, score in results:
        p = list(pv)
        groups[p[group_idx]].append((p[x_idx], score))

    plt.figure()
    for group_val in sorted(groups.keys(), key=lambda v: str(v)):
        pairs = groups[group_val]
        raw_x = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        x_enc, x_ticks = _encode_column(raw_x)
        order = np.argsort(x_enc)
        x_sorted = x_enc[order]
        y_sorted = np.array(scores)[order]
        label = _display_label(group_val)
        plt.plot(x_sorted, y_sorted, marker="o", label=f"{display_names[group_idx]}={label}")

    # Set x ticks using first group's encoding (all groups share the same x values)
    first_group = groups[sorted(groups.keys(), key=lambda v: str(v))[0]]
    _, x_ticks = _encode_column([p[0] for p in first_group])
    if x_ticks is not None:
        plt.xticks(range(len(x_ticks)), x_ticks, rotation=45, ha="right")

    plt.xlabel(display_names[x_idx])
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} by {display_names[group_idx]}")
    plt.legend(fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}{suffix}_lines.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


def _plot_3d_scatter(param_names, results, exp_path, metric_label, suffix):
    """
    3D scatter for exactly 3 params. Axes are the 3 params, color is the metric.
    Saved as interactive HTML.
    """
    display_names = [n.split(".")[-1] for n in param_names]

    cols = [[], [], []]
    scores = []
    for pv, score in results:
        p = list(pv)
        for i in range(3):
            cols[i].append(p[i])
        scores.append(score)

    encoded = []
    tick_configs = []
    for i in range(3):
        enc, ticks = _encode_column(cols[i])
        encoded.append(enc)
        tick_configs.append(ticks)

    score_array = np.array(scores, dtype=float)

    scene_axes = {}
    for i, axis_name in enumerate(["xaxis", "yaxis", "zaxis"]):
        axis_cfg = dict(title=display_names[i])
        if tick_configs[i] is not None:
            axis_cfg.update(dict(
                tickmode="array",
                tickvals=list(range(len(tick_configs[i]))),
                ticktext=tick_configs[i],
            ))
        scene_axes[axis_name] = axis_cfg

    # Build hover text
    hover_texts = []
    for idx in range(len(scores)):
        parts = []
        for i in range(3):
            val = encoded[i][idx]
            if tick_configs[i] is not None:
                i_val = int(round(val))
                if 0 <= i_val < len(tick_configs[i]):
                    label = tick_configs[i][i_val]
                else:
                    label = str(val)
            else:
                label = f"{val:g}"
            parts.append(f"{display_names[i]}: {label}")
        parts.append(f"{metric_label}: {score_array[idx]:.4f}")
        hover_texts.append("<br>".join(parts))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=encoded[0],
        y=encoded[1],
        z=encoded[2],
        mode="markers",
        text=hover_texts,
        hoverinfo="text",
        marker=dict(
            size=5,
            color=score_array,
            colorscale="Plasma",
            colorbar=dict(title=metric_label, len=0.6, x=-0.15),
            opacity=0.8,
        ),
    ))

    fig.update_layout(
        title_text=f"Grid Search (color = {metric_label})",
        scene=scene_axes,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}{suffix}.html"
    fig.write_html(filename)
    print(f"Interactive figure saved to {filename}")


def _plot_kq_gen_combined(param_names, summary_stats, exp_path):
    """
    If we're doing a 1D grid search and both `kernel_quality` and `generalization`
    exist, plot them on the same figure (KQ+Gen) and return the keys that should
    be skipped in the generic per-metric plotting loop.
    """
    if len(param_names) != 1:
        return set()

    display_name = param_names[0].split(".")[-1]

    kq_pairs = [
        (param_tuple, float(stats["kernel_quality"]))
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and ("kernel_quality" in stats)
    ]
    gen_pairs = [
        (param_tuple, float(stats["generalization"]))
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and ("generalization" in stats)
    ]

    if not kq_pairs or not gen_pairs:
        return set()

    x_kq, x_kq_ticks = _encode_column([list(pv)[0] for pv, _ in kq_pairs])
    y_kq = np.array([score for _, score in kq_pairs], dtype=float)

    x_gen, _ = _encode_column([list(pv)[0] for pv, _ in gen_pairs])
    y_gen = np.array([score for _, score in gen_pairs], dtype=float)

    order_kq = np.argsort(x_kq)
    x_kq, y_kq = x_kq[order_kq], y_kq[order_kq]
    order_gen = np.argsort(x_gen)
    x_gen, y_gen = x_gen[order_gen], y_gen[order_gen]

    plt.figure()
    plt.plot(x_kq, y_kq, marker="o", label="Kernel Quality", linewidth=2)
    plt.plot(x_gen, y_gen, marker="s", label="Generalization", linewidth=2)

    if x_kq_ticks is not None:
        plt.xticks(range(len(x_kq_ticks)), x_kq_ticks)

    plt.xlabel(display_name)
    plt.ylabel("Score")
    plt.title("Kernel Quality vs Generalization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{exp_path}/plot_{display_name}_kq_gen.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

    return {"kernel_quality", "generalization"}


def _plot_2d_kq_gen_heatmap(param_names, kq_results, gen_results, exp_path):
    """
    1x3 subplot heatmap: KQ, Gen, and KQ-Gen side by side.
    Each cell annotated with its own value only.
    """
    display_names = [n.split(".")[-1] for n in param_names]

    raw_x_kq = [list(pv)[0] for pv, _ in kq_results]
    raw_y_kq = [list(pv)[1] for pv, _ in kq_results]
    scores_kq = {(list(pv)[0], list(pv)[1]): s for pv, s in kq_results}
    scores_gen = {(list(pv)[0], list(pv)[1]): s for pv, s in gen_results}

    x_enc, x_ticks = _encode_column(raw_x_kq)
    y_enc, y_ticks = _encode_column(raw_y_kq)
    x_unique = np.unique(x_enc)
    y_unique = np.unique(y_enc)

    raw_x_vals = [list(pv)[0] for pv, _ in kq_results]
    raw_y_vals = [list(pv)[1] for pv, _ in kq_results]
    x_enc_to_raw = {}
    for xe, xr in zip(x_enc, raw_x_vals):
        x_enc_to_raw[xe] = xr
    y_enc_to_raw = {}
    for ye, yr in zip(y_enc, raw_y_vals):
        y_enc_to_raw[ye] = yr

    grid_kq = np.full((len(y_unique), len(x_unique)), np.nan)
    grid_gen = np.full((len(y_unique), len(x_unique)), np.nan)
    grid_diff = np.full((len(y_unique), len(x_unique)), np.nan)

    for j, yu in enumerate(y_unique):
        for i, xu in enumerate(x_unique):
            key = (x_enc_to_raw[xu], y_enc_to_raw[yu])
            kq_val = scores_kq.get(key)
            gen_val = scores_gen.get(key)
            if kq_val is not None:
                grid_kq[j, i] = kq_val
            if gen_val is not None:
                grid_gen[j, i] = gen_val
            if kq_val is not None and gen_val is not None:
                grid_diff[j, i] = kq_val - gen_val

    x_labels = x_ticks if x_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in x_unique]
    y_labels = y_ticks if y_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in y_unique]

    panels = [
        (grid_kq, "Kernel Quality"),
        (grid_gen, "Generalization"),
        (grid_diff, "KQ - Gen"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(6, len(x_unique) * 0.8 + 2) * 3, max(5, len(y_unique) * 0.6 + 2)))
    for ax, (grid, title) in zip(axes, panels):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="plasma")
        fig.colorbar(im, ax=ax, label=title)

        for j in range(len(y_unique)):
            for i in range(len(x_unique)):
                val = grid[j, i]
                if np.isfinite(val):
                    ax.text(i, j, f"{val:.4f}", ha="center", va="center", fontsize=7,
                            color="white" if val < (np.nanmin(grid) + np.nanmax(grid)) / 2 else "black")

        ax.set_xticks(range(len(x_unique)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(y_unique)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(display_names[0])
        ax.set_ylabel(display_names[1])
        ax.set_title(title)

    plt.tight_layout()

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}_kq_gen_heatmap.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


def _plot_2d_kq_gen_lines(param_names, kq_results, gen_results, exp_path):
    """
    Combined KQ/Gen line plot. KQ lines are solid, Gen lines are dashed.
    Same color for same param value. Max 3 line group values (6 lines total).
    """
    display_names = [n.split(".")[-1] for n in param_names]
    group_idx, x_idx, unique_counts = _select_line_group(kq_results)

    max_groups = 3
    if unique_counts[group_idx] > max_groups:
        print(f"Skipping KQ/Gen line plot: {display_names[group_idx]} has "
              f"{unique_counts[group_idx]} groups (max {max_groups})")
        return

    kq_groups = defaultdict(list)
    for pv, score in kq_results:
        p = list(pv)
        kq_groups[p[group_idx]].append((p[x_idx], score))

    gen_groups = defaultdict(list)
    for pv, score in gen_results:
        p = list(pv)
        gen_groups[p[group_idx]].append((p[x_idx], score))

    plt.figure()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, group_val in enumerate(sorted(kq_groups.keys(), key=lambda v: str(v))):
        color = colors[i % len(colors)]
        label_str = _display_label(group_val)

        # KQ line (solid)
        pairs = kq_groups[group_val]
        raw_x = [p[0] for p in pairs]
        scores = [p[1] for p in pairs]
        x_enc, x_ticks = _encode_column(raw_x)
        order = np.argsort(x_enc)
        plt.plot(x_enc[order], np.array(scores)[order], marker="o", linestyle="-",
                 color=color, label=f"KQ {display_names[group_idx]}={label_str}")

        # Gen line (dashed)
        if group_val in gen_groups:
            pairs = gen_groups[group_val]
            raw_x = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            x_enc, _ = _encode_column(raw_x)
            order = np.argsort(x_enc)
            plt.plot(x_enc[order], np.array(scores)[order], marker="s", linestyle="--",
                     color=color, label=f"Gen {display_names[group_idx]}={label_str}")

    # Set x ticks from first group
    first_group = kq_groups[sorted(kq_groups.keys(), key=lambda v: str(v))[0]]
    _, x_ticks = _encode_column([p[0] for p in first_group])
    if x_ticks is not None:
        plt.xticks(range(len(x_ticks)), x_ticks, rotation=45, ha="right")

    plt.xlabel(display_names[x_idx])
    plt.ylabel("Score")
    plt.title(f"KQ vs Gen by {display_names[group_idx]}")
    plt.legend(fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}_kq_gen_lines.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


def _plot_kq_gen_combined_2d(param_names, summary_stats, exp_path):
    """
    For 2D grid searches: produce combined KQ/Gen heatmap and line plot.
    Returns the set of metric keys to skip in the generic per-metric loop.
    """
    if len(param_names) != 2:
        return set()

    kq_results = [
        (param_tuple, float(stats["kernel_quality"]))
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "kernel_quality" in stats
    ]
    gen_results = [
        (param_tuple, float(stats["generalization"]))
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "generalization" in stats
    ]

    if not kq_results or not gen_results:
        return set()

    _plot_2d_kq_gen_heatmap(param_names, kq_results, gen_results, exp_path)
    _plot_2d_kq_gen_lines(param_names, kq_results, gen_results, exp_path)

    return {"kernel_quality", "generalization"}


def plot_gridsearch_results(
    param_names,
    summary_stats,
    exp_path,
    *,
    filter_threshold=0.8,
):
    """
    Plot any scalar stats present in `summary_stats`.

    Dispatches by number of grid-search parameters:
      1 param  -> 1D line plot (PNG)
      2 params -> heatmap + overlaid lines (PNG)
      3 params -> 3D scatter (HTML)
      >3 params -> skip with warning
    """
    if not summary_stats:
        return

    num_params = len(param_names)

    if num_params > 3:
        print(f"Warning: {num_params} grid-search params — plotting not supported for >3 params. Skipping.")
        return

    metric_specs = {
        "score": "NRMSE",
        "avg_node_variance": "Average Node Variance",
        "avg_node_mean": "Average Node Mean",
        "mean_spread": "Mean Spread",
        "avg_total_recurrent_influence": "Average Total Recurrent Influence",
        "kernel_quality": "Kernel Quality",
        "generalization": "Generalization",
        "memory_capacity": "Memory Capacity",
    }

    skip_keys = _plot_kq_gen_combined(param_names, summary_stats, exp_path)
    skip_keys |= _plot_kq_gen_combined_2d(param_names, summary_stats, exp_path)

    for stat_key, stat_label in metric_specs.items():
        if stat_key in skip_keys:
            continue

        stat_results = [
            (param_tuple, float(stats[stat_key]))
            for param_tuple, stats in summary_stats.items()
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and (stat_key in stats)
        ]

        if not stat_results:
            continue

        # Apply filter threshold only to NRMSE
        if stat_key == "score" and filter_threshold is not None:
            stat_results = [(pv, s) for pv, s in stat_results if s <= filter_threshold]
            if not stat_results:
                print(f"No results with {stat_label} <= {filter_threshold} to plot")
                continue

        suffix = "" if stat_key == "score" else f"_{stat_key}"

        if num_params == 1:
            display_name = param_names[0].split(".")[-1]
            raw_x = [list(pv)[0] for pv, _ in stat_results]
            x_enc, x_ticks = _encode_column(raw_x)
            y = np.array([s for _, s in stat_results], dtype=float)
            filepath = f"{exp_path}/plot_{display_name}{suffix}.png"
            _plot_1d(x_enc, y, x_ticks, display_name, stat_label,
                     f"Grid Search Performance ({stat_label})", filepath)

        elif num_params == 2:
            _plot_2d_heatmap(param_names, stat_results, exp_path, stat_label, suffix)
            _plot_2d_lines(param_names, stat_results, exp_path, stat_label, suffix)

        elif num_params == 3:
            _plot_3d_scatter(param_names, stat_results, exp_path, stat_label, suffix)
