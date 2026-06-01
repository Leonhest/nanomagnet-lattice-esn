import numpy as np
import matplotlib.pyplot as plt

# (label, NRMSE, std) per panel; CMA-ES points labeled by their TRAINING task
panels = {
    "narma10_to_mackey": [
        ("baseline-esn", 0.0677, 0.0428),
        ("constant", 0.0817, 0.0370),
        ("cmaes\n(NARMA-10)", 0.0223, 0.0162),   # transferred
        ("cmaes\n(Mackey)", 0.0022, 0.0000),     # target-trained
    ],
    "narma10_to_narma20": [
        ("baseline-esn", 0.3864, 0.0321),
        ("constant", 0.4764, 0.0305),
        ("cmaes\n(NARMA-10)", 0.3388, 0.0415),   # transferred
        ("cmaes\n(NARMA-20)", 0.2032, 0.0195),   # target-trained
    ],
}

OUT = "/Users/leonhesthaug/master/workspace/thesis/Figures/experiments/tiles/domain_adapt"

for name, rows in panels.items():
    labels = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    std = np.array([r[2] for r in rows])
    x = np.arange(len(rows))

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, y - std, y + std, alpha=0.2)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.xlabel("type")
    plt.ylabel("NRMSE")
    plt.title("Grid Search Performance (NRMSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"/tmp/{name}_plot_type_with_std.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote", out)
