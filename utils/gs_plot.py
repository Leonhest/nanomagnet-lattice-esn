import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def _display_label(val):
    """Shorten file paths to their parent folder name for plot labels."""
    if isinstance(val, str) and '/' in val and '.' in os.path.basename(val):
        return os.path.basename(os.path.dirname(val))
    return val


def _encode_params(results, num_params):
    """
    Helper to encode parameters into a numeric array for plotting.
    Returns:
        encoded_array: (N, num_params) float array
        axis_mappings: list of dicts or None. If axis_mappings[i] is not None,
                       it maps {str_value: int_index} for the i-th parameter.
        axis_tick_labels: list of lists or None. If axis_tick_labels[i] is not None,
                          it contains the string labels corresponding to indices.
    """
    raw_params = [list(pv) for pv, _ in results] # List of lists
    encoded_cols = []
    axis_mappings = []
    axis_tick_labels = []

    for col_idx in range(num_params):
        col_values = [row[col_idx] for row in raw_params]
        try:
            # Try converting to float
            col_float = np.array(col_values, dtype=float)
            encoded_cols.append(col_float)
            axis_mappings.append(None)
            axis_tick_labels.append(None)
        except (ValueError, TypeError):
            # If not numeric, encode as categorical
            unique_vals = sorted(list(set(col_values)))
            val_to_idx = {val: i for i, val in enumerate(unique_vals)}
            col_encoded = np.array([val_to_idx[v] for v in col_values], dtype=float)
            encoded_cols.append(col_encoded)
            axis_mappings.append(val_to_idx)
            axis_tick_labels.append([_display_label(v) for v in unique_vals])

    encoded_array = np.column_stack(encoded_cols)
    return encoded_array, axis_mappings, axis_tick_labels

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

    # Use encoding helper
    x_kq_encoded, x_kq_map, x_kq_labels = _encode_params(kq_pairs, 1)
    x_kq = x_kq_encoded[:, 0]
    y_kq = np.array([score for _, score in kq_pairs], dtype=float)
    
    x_gen_encoded, x_gen_map, x_gen_labels = _encode_params(gen_pairs, 1)
    x_gen = x_gen_encoded[:, 0]
    y_gen = np.array([score for _, score in gen_pairs], dtype=float)

    order_kq = np.argsort(x_kq)
    x_kq, y_kq = x_kq[order_kq], y_kq[order_kq]
    order_gen = np.argsort(x_gen)
    x_gen, y_gen = x_gen[order_gen], y_gen[order_gen]

    plt.figure()
    plt.plot(x_kq, y_kq, marker="o", label="Kernel Quality", linewidth=2)
    plt.plot(x_gen, y_gen, marker="s", label="Generalization", linewidth=2)
    
    # Handle categorical x-axis
    if x_kq_labels[0] is not None:
        plt.xticks(range(len(x_kq_labels[0])), x_kq_labels[0])
    
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

    # Use encoding helper to handle mixed types
    params_array, axis_mappings, axis_tick_labels = _encode_params(results, num_params)
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
        
        # Handle categorical x-axis
        if axis_tick_labels[0] is not None:
             plt.xticks(range(len(axis_tick_labels[0])), axis_tick_labels[0])

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
        
        # Update axis configuration for categorical data
        xaxis = dict(title=display_names[0])
        yaxis = dict(title=display_names[1])
        
        if axis_tick_labels[0] is not None:
            xaxis.update(dict(
                tickmode='array',
                tickvals=list(range(len(axis_tick_labels[0]))),
                ticktext=axis_tick_labels[0]
            ))
            
        if axis_tick_labels[1] is not None:
            yaxis.update(dict(
                tickmode='array',
                tickvals=list(range(len(axis_tick_labels[1]))),
                ticktext=axis_tick_labels[1]
            ))

        scene = dict(xaxis=xaxis, yaxis=yaxis, zaxis=dict(title=metric_label))
        title_text = f"Grid Search Performance ({metric_label})"
    else:  # 3 or more params
        x, y, z = params_array[:, 0], params_array[:, 1], params_array[:, 2]
        c = score_array
        
        xaxis = dict(title=display_names[0])
        yaxis = dict(title=display_names[1])
        zaxis = dict(title=display_names[2])
        
        if axis_tick_labels[0] is not None:
            xaxis.update(dict(tickmode='array', tickvals=list(range(len(axis_tick_labels[0]))), ticktext=axis_tick_labels[0]))
        if axis_tick_labels[1] is not None:
            yaxis.update(dict(tickmode='array', tickvals=list(range(len(axis_tick_labels[1]))), ticktext=axis_tick_labels[1]))
        if axis_tick_labels[2] is not None:
            zaxis.update(dict(tickmode='array', tickvals=list(range(len(axis_tick_labels[2]))), ticktext=axis_tick_labels[2]))

        scene = dict(xaxis=xaxis, yaxis=yaxis, zaxis=zaxis)
        title_text = f"Grid Search Performance (color = {metric_label})"
        if num_params > 3:
            title_text = f"Grid Search (>3 params) projected to first 3 (color = {metric_label})"

    # Helper to format hover text
    def format_val(val, idx):
        if axis_tick_labels[idx] is not None:
            # Map back from integer index to string label
            # Nearest integer
            i_val = int(round(val))
            if 0 <= i_val < len(axis_tick_labels[idx]):
                return axis_tick_labels[idx][i_val]
            return str(val)
        return f"{val}" # Numeric

    # We need custom hover data because simple %{x} will show the integer index
    # But Plotly's hovertemplate with 'array' tickmode usually shows the label automatically for axes.
    # However, for 3D scatter, it might show the coordinate. 
    # Let's rely on Plotly's behavior first, or we could construct a custom text array.
    
    # Construct custom text for hover to ensure labels are shown
    hover_texts = []
    for i in range(len(x)):
        x_str = format_val(x[i], 0)
        y_str = format_val(y[i], 1)
        
        if num_params == 2:
            z_val = z[i]
            hover_texts.append(f"{display_names[0]}: {x_str}<br>{display_names[1]}: {y_str}<br>{metric_label}: {z_val:.4f}")
        else:
            z_str = format_val(z[i], 2)
            c_val = c[i]
            hover_texts.append(f"{display_names[0]}: {x_str}<br>{display_names[1]}: {y_str}<br>{display_names[2]}: {z_str}<br>{metric_label}: {c_val:.4f}")

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
                    hovertemplate=f"{display_names[0]}: %{{x}}<br>{display_names[1]}: %{{y}}<br>{metric_label}: %{{z:.4f}}<extra></extra>",
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                text=hover_texts,
                hoverinfo="text", 
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
                text=hover_texts,
                hoverinfo="text",
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
    Plot any scalar stats present in `summary_stats`.

    Args:
        param_names: list of parameter names
        summary_stats: dict of summary statistics
        exp_path: path to save plots
        filter_threshold: optional threshold (applies only to the main score plot)
    """
    if not summary_stats:
        return

    metric_specs = {
        "score": "NRMSE",
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
    # We need to flatten this and then encode parameters if necessary
    
    flat_data = []
    for param_tuple, decile_values in decile_stats.items():
         for decile_idx, stat_value in enumerate(decile_values):
             flat_data.append((param_tuple, decile_idx, stat_value))
             
    if not flat_data:
        return

    # Extract param tuples for encoding
    param_tuples = [item[0] for item in flat_data]
    # Encode
    encoded_params, axis_mappings, axis_tick_labels = _encode_params([(pt, 0) for pt in param_tuples], len(param_names))
    
    data_points = []
    for i, (_, decile_idx, stat_value) in enumerate(flat_data):
        # Reconstruct data points with encoded values
        # Encoded params + decile + stat
        if len(param_names) == 1:
             data_points.append((encoded_params[i, 0], decile_idx, stat_value))
        else:
             data_points.append((encoded_params[i, 0], encoded_params[i, 1], decile_idx, stat_value))
    
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
        
        xaxis = dict(title=display_names[0])
        if axis_tick_labels[0] is not None:
             xaxis.update(dict(
                tickmode='array',
                tickvals=list(range(len(axis_tick_labels[0]))),
                ticktext=axis_tick_labels[0]
            ))

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
            xaxis=xaxis,
            yaxis=dict(title="Decile"),
            zaxis=dict(title=stat_label)
        )
        title_text = f'Decile Statistics: {stat_label}'
    else:
        # 3D plot: param1, param2, decile, color = stat value
        x, y, z, c = zip(*data_points)
        c_min, c_max = min(c), max(c)
        
        xaxis = dict(title=display_names[0])
        yaxis = dict(title=display_names[1])
        
        if axis_tick_labels[0] is not None:
             xaxis.update(dict(
                tickmode='array',
                tickvals=list(range(len(axis_tick_labels[0]))),
                ticktext=axis_tick_labels[0]
            ))
        if axis_tick_labels[1] is not None:
             yaxis.update(dict(
                tickmode='array',
                tickvals=list(range(len(axis_tick_labels[1]))),
                ticktext=axis_tick_labels[1]
            ))

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
            xaxis=xaxis,
            yaxis=yaxis,
            zaxis=dict(title="Decile")
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
