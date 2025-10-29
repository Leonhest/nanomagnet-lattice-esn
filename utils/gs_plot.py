import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_gridsearch_results(param_names, results):
    """
    Plot grid search results.
    """
    if not results:
        return

    # Filter out results with NRMSE > 0.8
    results = [(pv, score) for pv, score in results if score <= 0.8]
    
    if not results:
        print("No results with NRMSE <= 0.8 to plot")
        return

    display_names = [name.split('.')[-1] for name in param_names]
    num_params = len(param_names)
    params_array = np.array([list(pv) for pv, _ in results], dtype=float)
    nrmse_array = np.array([score for _, score in results], dtype=float)
    
    if num_params == 1:
        # Keep matplotlib for the simple 2D case
        x = params_array[:, 0]
        y = nrmse_array
        order = np.argsort(x)
        x, y = x[order], y[order]

        plt.figure()
        plt.plot(x, y, marker='o')
        plt.xlabel(display_names[0])
        plt.ylabel('NRMSE')
        plt.title('Grid Search Performance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Generate a descriptive filename and save the plot
        filename = f"experiments/plot_{display_names[0]}.png"
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
        z = c = nrmse_array # NRMSE is both z-axis and color
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title='NRMSE')
        title_text = 'Grid Search Performance'
    else: # 3 or more params
        x, y, z = params_array[:, 0], params_array[:, 1], params_array[:, 2]
        c = nrmse_array # NRMSE is color
        scene = dict(xaxis_title=display_names[0], yaxis_title=display_names[1], zaxis_title=display_names[2])
        title_text = 'Grid Search Performance (color = NRMSE)'
        if num_params > 3:
            title_text = 'Grid Search (>3 params) projected to first 3 (color = NRMSE)'

    # --- Create and save the plot ---
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=5,
            color=c,
            colorscale=custom_colorscale,
            cmin=nrmse_min,
            cmax=nrmse_max,
            colorbar=dict(title='NRMSE', len=0.6, x=-0.15),
            opacity=0.8
        )
    )])
    fig.update_layout(
        title_text=title_text,
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Generate a descriptive filename
    param_str = "_vs_".join(display_names)
    filename = f"experiments/plot_{param_str}.html"

    fig.write_html(filename)
    print(f"Interactive figure saved to {filename}")


