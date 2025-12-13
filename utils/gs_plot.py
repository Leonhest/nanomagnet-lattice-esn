import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

    x_kq = np.array([list(pv)[0] for pv, _ in kq_pairs], dtype=float)
    y_kq = np.array([score for _, score in kq_pairs], dtype=float)
    x_gen = np.array([list(pv)[0] for pv, _ in gen_pairs], dtype=float)
    y_gen = np.array([score for _, score in gen_pairs], dtype=float)

    order_kq = np.argsort(x_kq)
    x_kq, y_kq = x_kq[order_kq], y_kq[order_kq]
    order_gen = np.argsort(x_gen)
    x_gen, y_gen = x_gen[order_gen], y_gen[order_gen]

    plt.figure()
    plt.plot(x_kq, y_kq, marker="o", label="Kernel Quality", linewidth=2)
    plt.plot(x_gen, y_gen, marker="s", label="Generalization", linewidth=2)
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


def _plot_scalar_gridsearch_results(
    param_names,
    results,
    exp_path,
    *,
    metric_label="NRMSE",
    filter_threshold=None,
    filename_suffix=None,
):
    """
    Plot a single scalar metric across a grid search.

    Args:
        param_names: list[str]
        results: list[tuple[param_tuple, float]]
        exp_path: str
        metric_label: label for axis/title
        filter_threshold: if set, keep only points with value <= threshold (useful for NRMSE)
        filename_suffix: optional suffix appended to output filename
    """
    if not results:
        return

    display_names = [name.split(".")[-1] for name in param_names]
    num_params = len(param_names)

    if filter_threshold is not None:
        results = [(pv, score) for pv, score in results if score <= filter_threshold]
        if not results:
            print(f"No results with {metric_label} <= {filter_threshold} to plot")
            return

    params_array = np.array([list(pv) for pv, _ in results], dtype=float)
    score_array = np.array([score for _, score in results], dtype=float)

    suffix_part = ""
    if filename_suffix:
        safe_suffix = filename_suffix.strip().replace(" ", "_").replace("/", "_")
        if safe_suffix:
            suffix_part = f"_{safe_suffix}"

    if num_params == 1:
        x = params_array[:, 0]
        y = score_array
        order = np.argsort(x)
        x, y = x[order], y[order]

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel(display_names[0])
        plt.ylabel(metric_label)
        plt.title(f"Grid Search Performance ({metric_label})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{exp_path}/plot_{display_names[0]}{suffix_part}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close()
        return

    score_min = score_array.min()
    score_max = score_array.max()

    # Create a custom colorscale that gives more range to lower values
    plasma_colors = px.colors.sequential.Plasma
    n_stops = 20
    custom_colorscale = []
    for i in range(n_stops + 1):
        pos = i / n_stops
        pos_power = pos**0.5
        color_idx = int(pos_power * (len(plasma_colors) - 1))
        color_idx = max(min(color_idx, len(plasma_colors) - 1), 0)
        custom_colorscale.append([pos, plasma_colors[color_idx]])

    if num_params == 2:
        x, y = params_array[:, 0], params_array[:, 1]
        z = c = score_array
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title=metric_label)
        title_text = f"Grid Search Performance ({metric_label})"
    else:  # 3 or more params
        x, y, z = params_array[:, 0], params_array[:, 1], params_array[:, 2]
        c = score_array
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title=display_names[2])
        title_text = f"Grid Search Performance (color = {metric_label})"
        if num_params > 3:
            title_text = f"Grid Search (>3 params) projected to first 3 (color = {metric_label})"

    if num_params == 2:
        hovertemplate = (
            f"{display_names[0]}: %{{x}}<br>"
            f"{display_names[1]}: %{{y}}<br>"
            f"{metric_label}: %{{z:.4f}}<extra></extra>"
        )
    else:
        hovertemplate = (
            f"{display_names[0]}: %{{x}}<br>"
            f"{display_names[1]}: %{{y}}<br>"
            f"{display_names[2]}: %{{z}}<br>"
            f"{metric_label}: %{{customdata:.4f}}<extra></extra>"
        )

    fig = go.Figure()

    if num_params == 2:
        x_unique = np.unique(x)
        y_unique = np.unique(y)

        z_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        index_map = {(float(xv), float(yv)): zv for xv, yv, zv in zip(x, y, z)}
        for j, yv in enumerate(y_unique):
            for i, xv in enumerate(x_unique):
                if (float(xv), float(yv)) in index_map:
                    z_grid[j, i] = index_map[(float(xv), float(yv))]

        if x_unique.size >= 2 and y_unique.size >= 2 and np.isfinite(z_grid).sum() >= 4:
            fig.add_trace(
                go.Surface(
                    x=x_unique,
                    y=y_unique,
                    z=z_grid,
                    colorscale=custom_colorscale,
                    cmin=score_min,
                    cmax=score_max,
                    colorbar=dict(title=metric_label, len=0.6, x=-0.15),
                    showscale=True,
                    opacity=0.8,
                    hovertemplate=hovertemplate,
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                customdata=c,
                hovertemplate=hovertemplate,
                marker=dict(
                    size=4,
                    color=c,
                    colorscale=custom_colorscale,
                    cmin=score_min,
                    cmax=score_max,
                    opacity=0.9,
                ),
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                customdata=c,
                hovertemplate=hovertemplate,
                marker=dict(
                    size=5,
                    color=c,
                    colorscale=custom_colorscale,
                    cmin=score_min,
                    cmax=score_max,
                    colorbar=dict(title=metric_label, len=0.6, x=-0.15),
                    opacity=0.8,
                ),
            )
        )

    fig.update_layout(title_text=title_text, scene=scene, margin=dict(l=0, r=0, b=0, t=40))

    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}{suffix_part}.html"
    fig.write_html(filename)
    print(f"Interactive figure saved to {filename}")


def plot_gridsearch_results(
    param_names,
    summary_stats,
    exp_path,
    *,
    filter_threshold=0.8,
):
    """
    Plot the main score (`stats['score']`) plus any known extra scalar stats present
    in `summary_stats`.

    Args:
        param_names: list[str]
        summary_stats: dict[param_tuple, dict[str, Any]]
        exp_path: str
        metric_label: label for the main score
        filter_threshold: optional threshold (applies only to the main score plot)
    """
    if not summary_stats:
        return

    display_names = [name.split(".")[-1] for name in param_names]
    num_params = len(param_names)

    # Plot known scalar fields (avoid huge arrays like node_means).
    metric_specs = {
        # Primary score (NRMSE runs only)
        "score": "NRMSE",
        # Reservoir summary stats
        "avg_node_variance": "Average Node Variance",
        "avg_node_mean": "Average Node Mean",
        "mean_spread": "Mean Spread",
        "avg_total_recurrent_influence": "Average Total Recurrent Influence",
        # res_metrics summary stats
        "kernel_quality": "Kernel Quality",
        "generalization": "Generalization",
        "memory_capacity": "Memory Capacity",
    }

    skip_keys = _plot_kq_gen_combined(param_names, summary_stats, exp_path)

    for stat_key, stat_label in metric_specs.items():
        if stat_key in skip_keys:
            continue
        stat_results = [
            (param_tuple, float(stats[stat_key]))
            for param_tuple, stats in summary_stats.items()
            if isinstance(param_tuple, tuple) and isinstance(stats, dict) and (stat_key in stats)
        ]
        _plot_scalar_gridsearch_results(
            param_names,
            stat_results,
            exp_path,
            metric_label=stat_label,
            filter_threshold=filter_threshold if stat_key == "score" else None,
            filename_suffix=None if stat_key == "score" else stat_key,
        )


def plot_decile_statistics(param_names, decile_stats, exp_path, stat_key, stat_label):
    """
    Plot decile statistics. Works with 0, 1, or 2 gridsearch parameters.
    - 0 params: 2D plot (decile vs statistic value)
    - 1 param: 3D surface plot (param vs decile, color = statistic value)
    - 2 params: 3D scatter plot (param1 vs param2 vs decile, color = statistic value)
    
    Args:
        param_names: List of parameter names (max 2)
        decile_stats: Dict mapping (param_tuple) -> list of 10 averaged decile stat values
        exp_path: Path to save plots
        stat_key: Key for the statistic ('avg_node_mean', 'avg_node_variance', 'mean_spread')
        stat_label: Display label for the statistic
    """
    if len(param_names) > 2:
        print(f"Cannot plot deciles with >2 parameters. Skipping {stat_key}.")
        return
    
    display_names = [name.split('.')[-1] for name in param_names]
    
    # Handle case with 0 gridsearch parameters - simple 2D plot
    if len(param_names) == 0:
        # There should be only one entry with empty tuple key
        if not decile_stats:
            return
        # Get the first (and likely only) decile values
        decile_values = list(decile_stats.values())[0]
        deciles = np.arange(10)
        values = np.array(decile_values)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=deciles,
            y=values,
            mode='lines+markers',
            marker=dict(size=8),
            line=dict(width=2)
        ))
        fig.update_layout(
            title_text=f'Decile Statistics: {stat_label}',
            xaxis_title="Decile",
            yaxis_title=stat_label,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        filename = f"{exp_path}/plot_decile_{stat_key}.html"
        fig.write_html(filename)
        print(f"Decile plot saved to {filename}")
        return
    
    # Prepare data: (param1, param2, decile, stat_value)
    data_points = []
    for param_tuple, decile_values in decile_stats.items():
        if len(param_names) == 1:
            x_val = param_tuple[0]
            y_val = None
        else:
            x_val = param_tuple[0]
            y_val = param_tuple[1]
        
        for decile_idx, stat_value in enumerate(decile_values):
            if len(param_names) == 1:
                # 2D plot: param vs decile
                data_points.append((x_val, decile_idx, stat_value))
            else:
                # 3D plot: param1, param2, decile
                data_points.append((x_val, y_val, decile_idx, stat_value))
    
    if not data_points:
        return
    
    # Create colorscale
    plasma_colors = px.colors.sequential.Plasma
    n_stops = 20
    custom_colorscale = []
    for i in range(n_stops + 1):
        pos = i / n_stops
        pos_power = pos ** 0.5
        color_idx = int(pos_power * (len(plasma_colors) - 1))
        color_idx = max(min(color_idx, len(plasma_colors) - 1), 0)
        custom_colorscale.append([pos, plasma_colors[color_idx]])
    
    fig = go.Figure()
    
    if len(param_names) == 1:
        # 2D plot: param vs decile, color = stat value
        x, y, z = zip(*data_points)
        z_min, z_max = min(z), max(z)
        
        # Try to form a regular grid
        x_unique = np.unique(x)
        y_unique = np.arange(10)  # Deciles 0-9
        
        z_grid = np.full((len(y_unique), len(x_unique)), np.nan, dtype=float)
        index_map = {(float(xv), int(yv)): zv for xv, yv, zv in zip(x, y, z)}
        for j, yv in enumerate(y_unique):
            for i, xv in enumerate(x_unique):
                if (float(xv), int(yv)) in index_map:
                    z_grid[j, i] = index_map[(float(xv), int(yv))]
        
        # Add surface if we have valid values
        if np.isfinite(z_grid).sum() >= 4:
            fig.add_trace(go.Surface(
                x=x_unique,
                y=y_unique,
                z=z_grid,
                colorscale=custom_colorscale,
                cmin=z_min,
                cmax=z_max,
                colorbar=dict(title=stat_label, len=0.6, x=-0.15),
                showscale=True,
                opacity=0.8,
                hovertemplate=(
                    f"{display_names[0]}: %{{x}}<br>"
                    f"Decile: %{{y}}<br>"
                    f"{stat_label}: %{{z:.4f}}<extra></extra>"
                )
            ))
        
        # Overlay scatter markers
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z), mode='markers',
            hovertemplate=(
                f"{display_names[0]}: %{{x}}<br>"
                f"Decile: %{{y}}<br>"
                f"{stat_label}: %{{z:.4f}}<extra></extra>"
            ),
            marker=dict(
                size=4,
                color=list(z),
                colorscale=custom_colorscale,
                cmin=z_min,
                cmax=z_max,
                opacity=0.9
            ),
            showlegend=False
        ))
        
        scene = dict(
            xaxis_title=display_names[0],
            yaxis_title="Decile",
            zaxis_title=stat_label
        )
        title_text = f'Decile Statistics: {stat_label}'
    else:
        # 3D plot: param1, param2, decile, color = stat value
        x, y, z, c = zip(*data_points)
        c_min, c_max = min(c), max(c)
        
        fig.add_trace(go.Scatter3d(
            x=list(x), y=list(y), z=list(z), mode='markers',
            customdata=list(c),
            hovertemplate=(
                f"{display_names[0]}: %{{x}}<br>"
                f"{display_names[1]}: %{{y}}<br>"
                f"Decile: %{{z}}<br>"
                f"{stat_label}: %{{customdata:.4f}}<extra></extra>"
            ),
            marker=dict(
                size=5,
                color=list(c),
                colorscale=custom_colorscale,
                cmin=c_min,
                cmax=c_max,
                colorbar=dict(title=stat_label, len=0.6, x=-0.15),
                opacity=0.8
            )
        ))
        
        scene = dict(
            xaxis_title=display_names[0],
            yaxis_title=display_names[1],
            zaxis_title="Decile"
        )
        title_text = f'Decile Statistics: {stat_label} (color)'
    
    fig.update_layout(
        title_text=title_text,
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Generate filename
    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}_decile_{stat_key}.html"
    
    fig.write_html(filename)
    print(f"Decile plot saved to {filename}")


