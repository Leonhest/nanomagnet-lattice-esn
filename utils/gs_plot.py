import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def _display_label(val):
    """Shorten file/directory paths to their basename for plot labels."""
    if isinstance(val, str) and '/' in val:
        # Strip trailing slash for directories, then take basename
        name = os.path.basename(os.path.normpath(val))
        # Only strip real file extensions (e.g. .json), not decimals like .0
        ext = os.path.splitext(name)[1]
        if ext and ext[1:].isalpha():
            name = os.path.splitext(name)[0]
        # Strip "best_tiles_" or "best_tile_" prefix
        for prefix in ("best_tiles_", "best_tile_", "best_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
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
        import re
        def _natural_sort_key(v):
            """Sort strings with embedded numbers in natural order."""
            return [int(c) if c.isdigit() else c.lower()
                    for c in re.split(r'(\d+)', str(v))]
        unique_vals = sorted(set(values), key=_natural_sort_key)
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        encoded = np.array([val_to_idx[v] for v in values], dtype=float)
        tick_labels = [_display_label(v) for v in unique_vals]
        return encoded, tick_labels


def _plot_1d(x, y, tick_labels, x_label, y_label, title, filepath, *, std=None):
    """Plot a 1D line chart and save as PNG."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    if std is not None:
        std = std[order]

    plt.figure()
    plt.plot(x, y, marker="o")
    if std is not None:
        plt.fill_between(x, y - std, y + std, alpha=0.2)
    if tick_labels is not None:
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
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


def _plot_2d_heatmap(param_names, results, exp_path, metric_label, suffix, *, std_map=None):
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
    std_grid = np.full((len(y_unique), len(x_unique)), np.nan) if std_map else None
    x_idx_map = {v: i for i, v in enumerate(x_unique)}
    y_idx_map = {v: i for i, v in enumerate(y_unique)}

    for idx, (xi, yi, s) in enumerate(zip(x_enc, y_enc, scores)):
        grid[y_idx_map[yi], x_idx_map[xi]] = s
        if std_map:
            pv = results[idx][0]
            if pv in std_map:
                std_grid[y_idx_map[yi], x_idx_map[xi]] = std_map[pv]

    fig, ax = plt.subplots(figsize=(max(6, len(x_unique) * 0.8 + 2), max(5, len(y_unique) * 0.6 + 2)))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="plasma")
    fig.colorbar(im, ax=ax, label=metric_label)

    # Annotate cells
    for j in range(len(y_unique)):
        for i in range(len(x_unique)):
            val = grid[j, i]
            if np.isfinite(val):
                text = f"{val:.4f}"
                if std_grid is not None and np.isfinite(std_grid[j, i]):
                    text += f"\n\u00b1{std_grid[j, i]:.4f}"
                ax.text(i, j, text, ha="center", va="center", fontsize=6,
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


def _plot_2d_lines(param_names, results, exp_path, metric_label, suffix, *, max_groups=6, std_map=None):
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
    std_groups = defaultdict(list) if std_map else None
    for pv, score in results:
        p = list(pv)
        groups[p[group_idx]].append((p[x_idx], score))
        if std_map and pv in std_map:
            std_groups[p[group_idx]].append((p[x_idx], std_map[pv]))

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

        if std_groups and group_val in std_groups:
            std_pairs = std_groups[group_val]
            std_vals = np.array([p[1] for p in std_pairs])[order]
            plt.fill_between(x_sorted, y_sorted - std_vals, y_sorted + std_vals, alpha=0.15)

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


def _plot_kq_gen_combined(param_names, summary_stats, exp_path, *, include_std=False):
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

    if include_std:
        kq_std = np.array([
            float(stats.get("kernel_quality_std", 0))
            for param_tuple, stats in summary_stats.items()
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "kernel_quality" in stats
        ], dtype=float)[order_kq]
        gen_std = np.array([
            float(stats.get("generalization_std", 0))
            for param_tuple, stats in summary_stats.items()
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "generalization" in stats
        ], dtype=float)[order_gen]
        plt.fill_between(x_kq, y_kq - kq_std, y_kq + kq_std, alpha=0.2)
        plt.fill_between(x_gen, y_gen - gen_std, y_gen + gen_std, alpha=0.2)

    if x_kq_ticks is not None:
        plt.xticks(range(len(x_kq_ticks)), x_kq_ticks)

    plt.xlabel(display_name)
    plt.ylabel("Score")
    plt.title("Kernel Quality vs Generalization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    suffix = "_with_std" if include_std else ""
    filename = f"{exp_path}/plot_{display_name}_kq_gen{suffix}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

    return {"kernel_quality", "generalization"}


def _plot_2d_kq_gen_heatmap(param_names, kq_results, gen_results, exp_path, *,
                            kq_std_map=None, gen_std_map=None):
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
    std_grid_kq = np.full((len(y_unique), len(x_unique)), np.nan) if kq_std_map else None
    std_grid_gen = np.full((len(y_unique), len(x_unique)), np.nan) if gen_std_map else None

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
            # Look up std by param_tuple
            if kq_std_map:
                for pt, std_val in kq_std_map.items():
                    pt_key = (list(pt)[0], list(pt)[1])
                    if pt_key == key:
                        std_grid_kq[j, i] = std_val
                        break
            if gen_std_map:
                for pt, std_val in gen_std_map.items():
                    pt_key = (list(pt)[0], list(pt)[1])
                    if pt_key == key:
                        std_grid_gen[j, i] = std_val
                        break

    x_labels = x_ticks if x_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in x_unique]
    y_labels = y_ticks if y_ticks is not None else [f"{v:g}" if v == int(v) else f"{v}" for v in y_unique]

    panels = [
        (grid_kq, "Kernel Quality", std_grid_kq),
        (grid_gen, "Generalization", std_grid_gen),
        (grid_diff, "KQ - Gen", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(max(6, len(x_unique) * 0.8 + 2) * 3, max(5, len(y_unique) * 0.6 + 2)))
    for ax, (grid, title, std_grid) in zip(axes, panels):
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="plasma")
        fig.colorbar(im, ax=ax, label=title)

        for j in range(len(y_unique)):
            for i in range(len(x_unique)):
                val = grid[j, i]
                if np.isfinite(val):
                    text = f"{val:.4f}"
                    if std_grid is not None and np.isfinite(std_grid[j, i]):
                        text += f"\n\u00b1{std_grid[j, i]:.4f}"
                    ax.text(i, j, text, ha="center", va="center", fontsize=6,
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
    suffix = "_with_std" if (kq_std_map or gen_std_map) else ""
    filename = f"{exp_path}/plot_{param_str}_kq_gen{suffix}_heatmap.png"
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")
    plt.close()


def _plot_2d_kq_gen_lines(param_names, kq_results, gen_results, exp_path, *,
                          kq_std_map=None, gen_std_map=None):
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
    kq_std_groups = defaultdict(list) if kq_std_map else None
    for pv, score in kq_results:
        p = list(pv)
        kq_groups[p[group_idx]].append((p[x_idx], score))
        if kq_std_map and pv in kq_std_map:
            kq_std_groups[p[group_idx]].append((p[x_idx], kq_std_map[pv]))

    gen_groups = defaultdict(list)
    gen_std_groups = defaultdict(list) if gen_std_map else None
    for pv, score in gen_results:
        p = list(pv)
        gen_groups[p[group_idx]].append((p[x_idx], score))
        if gen_std_map and pv in gen_std_map:
            gen_std_groups[p[group_idx]].append((p[x_idx], gen_std_map[pv]))

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
        x_sorted = x_enc[order]
        y_sorted = np.array(scores)[order]
        plt.plot(x_sorted, y_sorted, marker="o", linestyle="-",
                 color=color, label=f"KQ {display_names[group_idx]}={label_str}")
        if kq_std_groups and group_val in kq_std_groups:
            std_vals = np.array([p[1] for p in kq_std_groups[group_val]])[order]
            plt.fill_between(x_sorted, y_sorted - std_vals, y_sorted + std_vals, alpha=0.15, color=color)

        # Gen line (dashed)
        if group_val in gen_groups:
            pairs = gen_groups[group_val]
            raw_x = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            x_enc, _ = _encode_column(raw_x)
            order = np.argsort(x_enc)
            x_sorted = x_enc[order]
            y_sorted = np.array(scores)[order]
            plt.plot(x_sorted, y_sorted, marker="s", linestyle="--",
                     color=color, label=f"Gen {display_names[group_idx]}={label_str}")
            if gen_std_groups and group_val in gen_std_groups:
                std_vals = np.array([p[1] for p in gen_std_groups[group_val]])[order]
                plt.fill_between(x_sorted, y_sorted - std_vals, y_sorted + std_vals, alpha=0.15, color=color)

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
    suffix = "_with_std" if (kq_std_map or gen_std_map) else ""
    filename = f"{exp_path}/plot_{param_str}_kq_gen{suffix}_lines.png"
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

    # Without std
    _plot_2d_kq_gen_heatmap(param_names, kq_results, gen_results, exp_path)
    _plot_2d_kq_gen_lines(param_names, kq_results, gen_results, exp_path)

    # With std
    kq_std_map = {
        param_tuple: float(stats["kernel_quality_std"])
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "kernel_quality_std" in stats
    }
    gen_std_map = {
        param_tuple: float(stats["generalization_std"])
        for param_tuple, stats in summary_stats.items()
        if isinstance(param_tuple, tuple) and isinstance(stats, dict) and "generalization_std" in stats
    }
    if kq_std_map or gen_std_map:
        _plot_2d_kq_gen_heatmap(param_names, kq_results, gen_results, exp_path,
                                kq_std_map=kq_std_map, gen_std_map=gen_std_map)
        _plot_2d_kq_gen_lines(param_names, kq_results, gen_results, exp_path,
                              kq_std_map=kq_std_map, gen_std_map=gen_std_map)

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
        "avg_tri": "Average TRI",
        "avg_tri_ratio": "Average TRI Ratio",
        "orthogonality_error": "Orthogonality Error",
        "kernel_quality": "Kernel Quality",
        "generalization": "Generalization",
        "memory_capacity": "Memory Capacity",
    }

    skip_keys = _plot_kq_gen_combined(param_names, summary_stats, exp_path)
    _plot_kq_gen_combined(param_names, summary_stats, exp_path, include_std=True)
    skip_keys |= _plot_kq_gen_combined_2d(param_names, summary_stats, exp_path)

    for stat_key, stat_label in metric_specs.items():
        if stat_key in skip_keys:
            continue

        std_key = f"{stat_key}_std"
        stat_results = [
            (param_tuple, float(stats[stat_key]))
            for param_tuple, stats in summary_stats.items()
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and (stat_key in stats)
        ]

        if not stat_results:
            continue

        # Build std map for metrics that have std
        std_map = {}
        for param_tuple, stats in summary_stats.items():
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and std_key in stats:
                std_map[param_tuple] = float(stats[std_key])
        has_std = bool(std_map)

        # Apply filter threshold only to NRMSE
        if stat_key == "score" and filter_threshold is not None:
            stat_results = [(pv, s) for pv, s in stat_results if s <= filter_threshold]
            if has_std:
                std_map = {pv: s for pv, s in std_map.items() if pv in {r[0] for r in stat_results}}
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

            if has_std:
                std_arr = np.array([std_map.get(pv, 0.0) for pv, _ in stat_results], dtype=float)
                filepath_std = f"{exp_path}/plot_{display_name}{suffix}_with_std.png"
                _plot_1d(x_enc, y, x_ticks, display_name, stat_label,
                         f"Grid Search Performance ({stat_label})", filepath_std, std=std_arr)

        elif num_params == 2:
            _plot_2d_heatmap(param_names, stat_results, exp_path, stat_label, suffix)
            _plot_2d_lines(param_names, stat_results, exp_path, stat_label, suffix)

            if has_std:
                _plot_2d_heatmap(param_names, stat_results, exp_path, stat_label,
                                suffix + "_with_std", std_map=std_map)
                _plot_2d_lines(param_names, stat_results, exp_path, stat_label,
                               suffix + "_with_std", std_map=std_map)

        elif num_params == 3:
            _plot_3d_scatter(param_names, stat_results, exp_path, stat_label, suffix)
