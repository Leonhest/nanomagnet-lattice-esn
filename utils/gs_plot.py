import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection
import numpy as np


def plot_gridsearch_results(param_names, results):
    """
    Plot grid search results.

    Args:
        param_names (list[str]): Ordered list of parameter paths (dot notation) that varied.
        results (list[tuple]): List of (param_values_tuple, nrmse_float).
            - param_values_tuple aligns with param_names order
            - nrmse_float is the performance metric (lower is better)
    """
    if not results:
        return

    # Extract just the last part of each parameter name (after last dot)
    display_names = [name.split('.')[-1] for name in param_names]
    
    num_params = len(param_names)

    # Unpack results
    params_array = np.array([list(pv) for pv, _ in results], dtype=float)
    nrmse_array = np.array([score for _, score in results], dtype=float)

    if num_params == 1:
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
        plt.show()
        return

    if num_params == 2:
        x = params_array[:, 0]
        y = params_array[:, 1]
        z = nrmse_array

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=z, cmap='viridis')
        ax.set_xlabel(display_names[0])
        ax.set_ylabel(display_names[1])
        ax.set_zlabel('NRMSE')
        ax.set_title('Grid Search Performance')
        fig.colorbar(sc, ax=ax, shrink=0.6, label='NRMSE', location='left')
        plt.tight_layout()
        plt.show()
        return

    if num_params == 3:
        x = params_array[:, 0]
        y = params_array[:, 1]
        z = params_array[:, 2]
        c = nrmse_array

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x, y, z, c=c, cmap='viridis')
        ax.set_xlabel(display_names[0])
        ax.set_ylabel(display_names[1])
        ax.set_zlabel(display_names[2])
        ax.set_title('Grid Search Performance (color = NRMSE)')
        fig.colorbar(sc, ax=ax, shrink=0.6, label='NRMSE', location='left')
        plt.tight_layout()
        plt.show()
        return

    # >3 parameters: project to first 3 and color by NRMS E
    x = params_array[:, 0]
    y = params_array[:, 1]
    z = params_array[:, 2]
    c = nrmse_array

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=c, cmap='viridis')
    ax.set_xlabel(display_names[0])
    ax.set_ylabel(display_names[1])
    ax.set_zlabel(display_names[2])
    ax.set_title('Grid Search (>3 params) projected to first 3 (color = NRMSE)')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='NRMSE')
    plt.tight_layout()
    plt.show()


