import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_gridsearch_results(
    param_names,
    results,
    exp_path,
    res_metrics_mode=False,
    metrics_results=None,
    metric_label="NRMSE",
    filter_scores=True,
    filename_suffix=None,
):
    """
    Plot grid search results.
    """
    if not results:
        return

    display_names = [name.split('.')[-1] for name in param_names]
    num_params = len(param_names)
    
    # Handle res_metrics mode with 1 parameter - create two 2D plots
    if res_metrics_mode and num_params == 1 and metrics_results:
        # First plot: kernel_quality and generalization
        kq_results = metrics_results["kernel_quality"]
        gen_results = metrics_results["generalization"]
        
        # Extract x and y values
        x_kq = np.array([list(pv)[0] for pv, _ in kq_results], dtype=float)
        y_kq = np.array([score for _, score in kq_results], dtype=float)
        x_gen = np.array([list(pv)[0] for pv, _ in gen_results], dtype=float)
        y_gen = np.array([score for _, score in gen_results], dtype=float)
        
        # Sort by x values
        order_kq = np.argsort(x_kq)
        x_kq, y_kq = x_kq[order_kq], y_kq[order_kq]
        order_gen = np.argsort(x_gen)
        x_gen, y_gen = x_gen[order_gen], y_gen[order_gen]
        
        # Create first plot with both lines
        plt.figure()
        plt.plot(x_kq, y_kq, marker='o', label='Kernel Quality', linewidth=2)
        plt.plot(x_gen, y_gen, marker='s', label='Generalization', linewidth=2)
        plt.xlabel(display_names[0])
        plt.ylabel('Score')
        plt.title('Kernel Quality vs Generalization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename1 = f"{exp_path}/plot_{display_names[0]}_kq_gen.png"
        plt.savefig(filename1)
        print(f"Plot saved to {filename1}")
        plt.close()
        
        # Second plot: memory_capacity
        mc_results = metrics_results["memory_capacity"]
        x_mc = np.array([list(pv)[0] for pv, _ in mc_results], dtype=float)
        y_mc = np.array([score for _, score in mc_results], dtype=float)
        
        order_mc = np.argsort(x_mc)
        x_mc, y_mc = x_mc[order_mc], y_mc[order_mc]
        
        plt.figure()
        plt.plot(x_mc, y_mc, marker='o', linewidth=2)
        plt.xlabel(display_names[0])
        plt.ylabel('Memory Capacity')
        plt.title('Memory Capacity')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename2 = f"{exp_path}/plot_{display_names[0]}_memory_capacity.png"
        plt.savefig(filename2)
        print(f"Plot saved to {filename2}")
        plt.close()
        return
    
    # Filter out high scores when requested (default for NRMSE)
    if filter_scores and not res_metrics_mode:
        results = [(pv, score) for pv, score in results if score <= 0.8]
        
        if not results:
            print("No results with NRMSE <= 0.8 to plot")
            return

    params_array = np.array([list(pv) for pv, _ in results], dtype=float)
    nrmse_array = np.array([score for _, score in results], dtype=float)
    
    suffix_part = ""
    if filename_suffix:
        safe_suffix = filename_suffix.strip().replace(" ", "_").replace("/", "_")
        if safe_suffix:
            suffix_part = f"_{safe_suffix}"

    if num_params == 1:
        # Keep matplotlib for the simple 2D case
        x = params_array[:, 0]
        y = nrmse_array
        order = np.argsort(x)
        x, y = x[order], y[order]

        plt.figure()
        plt.plot(x, y, marker='o')
        plt.xlabel(display_names[0])
        plt.ylabel(metric_label)
        plt.title(f'Grid Search Performance ({metric_label})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Generate a descriptive filename and save the plot
        filename = f"{exp_path}/plot_{display_names[0]}{suffix_part}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.close() # Close the figure to prevent showing it interactively
        return

    # --- Common setup for all 3D plotly plots (num_params >= 2) ---
    nrmse_min = nrmse_array.min()
    nrmse_max = nrmse_array.max()

    # Create a custom colorscale that gives more range to lower values
    plasma_colors = px.colors.sequential.Plasma
    n_stops = 20
    custom_colorscale = []
    for i in range(n_stops + 1):
        pos = i / n_stops
        pos_power = pos ** 0.5
        color_idx = int(pos_power * (len(plasma_colors) - 1))
        color_idx = max(min(color_idx, len(plasma_colors) - 1), 0)
        custom_colorscale.append([pos, plasma_colors[color_idx]])

    # --- Configure plot based on number of parameters ---
    if num_params == 2:
        x, y = params_array[:, 0], params_array[:, 1]
        z = c = nrmse_array
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title=metric_label)
        title_text = f'Grid Search Performance ({metric_label})'
        z = c = nrmse_array
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title='NRMSE')
        title_text = 'Grid Search Performance'
    else: # 3 or more params
        x, y, z = params_array[:, 0], params_array[:, 1], params_array[:, 2]
        c = nrmse_array # NRMSE is color
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title=display_names[2])
        title_text = f'Grid Search Performance (color = {metric_label})'
        if num_params > 3:
            title_text = 'Grid Search (>3 params) projected to first 3 (color = NRMSE)'

    # --- Create and save the plot ---
    # Build hover template to include NR MSE explicitly
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

    # Build figure
    fig = go.Figure()

    if num_params == 2:
        # Try to form a regular grid to draw a surface/plane
        x_unique = np.unique(x)
        y_unique = np.unique(y)

        # Map (x,y) -> z and populate Z grid
        z_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        index_map = {(float(xv), float(yv)): zv for xv, yv, zv in zip(x, y, z)}
        for j, yv in enumerate(y_unique):
            for i, xv in enumerate(x_unique):
                if (float(xv), float(yv)) in index_map:
                    z_grid[j, i] = index_map[(float(xv), float(yv))]

        # Add surface if we have at least a 2x2 grid with some valid values
        if x_unique.size >= 2 and y_unique.size >= 2 and np.isfinite(z_grid).sum() >= 4:
            fig.add_trace(go.Surface(
                x=x_unique,
                y=y_unique,
                z=z_grid,
                colorscale=custom_colorscale,
                cmin=nrmse_min,
                cmax=nrmse_max,
                colorbar=dict(title=metric_label, len=0.6, x=-0.15),
                showscale=True,
                opacity=0.8,
                hovertemplate=(
                    f"{display_names[0]}: %{{x}}<br>"
                    f"{display_names[1]}: %{{y}}<br>"
                    f"{metric_label}: %{{z:.4f}}<extra></extra>"
                )
            ))

        # Overlay scatter markers for exact evaluated points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            customdata=c,
            hovertemplate=hovertemplate,
            marker=dict(
                size=4,
                color=c,
                colorscale=custom_colorscale,
                cmin=nrmse_min,
                cmax=nrmse_max,
                opacity=0.9
            ),
            showlegend=False
        ))
    else:
        # For >=3 params keep scatter3d
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            customdata=c,
            hovertemplate=hovertemplate,
            marker=dict(
                size=5,
                color=c,
                colorscale=custom_colorscale,
                cmin=nrmse_min,
                cmax=nrmse_max,
                colorbar=dict(title=metric_label, len=0.6, x=-0.15),
                opacity=0.8
            )
        ))
    fig.update_layout(
        title_text=title_text,
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Generate a descriptive filename
    param_str = "_vs_".join(display_names)
    filename = f"{exp_path}/plot_{param_str}{suffix_part}.html"

    fig.write_html(filename)
    print(f"Interactive figure saved to {filename}")


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


