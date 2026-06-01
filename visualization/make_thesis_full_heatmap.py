import json, ast, re
import numpy as np
import matplotlib.pyplot as plt

REPO = "/Users/leonhesthaug/master/workspace/nanomagnet-lattice-esn"

# --- CMA-ES constraint-flexibility grid (trained x deploy self-connection) ---
cm = json.load(open(f"{REPO}/experiments/v2/tiles/self_connections/reservoir_stats_summary.json"))
cma = {}
for k, v in cm.items():
    path, deploy = ast.literal_eval(k)
    trained = float(re.search(r"self_connection=([0-9.]+)\.json", path).group(1))
    cma[(trained, float(deploy))] = (v["score"], v["score_std"])

# --- unoptimized constant lattice vs (deploy) self-connection ---
sc = json.load(open(f"{REPO}/experiments/v2/self_connections/narma10/reservoir_stats_summary.json"))
const = {}
for k, v in sc.items():
    t, s = ast.literal_eval(k)
    if str(t).lower() == "constant":
        const[float(s)] = (v["score"], v["score_std"])

trained_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
deploy_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
ncols = len(trained_vals) + 1   # + constant column
nrows = len(deploy_vals)

grid = np.full((nrows, ncols), np.nan)
std = np.full((nrows, ncols), np.nan)
for di, d in enumerate(deploy_vals):
    for ti, t in enumerate(trained_vals):
        grid[di, ti], std[di, ti] = cma[(t, d)]
    grid[di, -1], std[di, -1] = const[d]     # constant column on the right

vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))   # 0.0858 .. 0.4247

fig, ax = plt.subplots(figsize=(ncols * 0.8 + 2, nrows * 0.6 + 2))
im = ax.imshow(grid, aspect="auto", origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)
fig.colorbar(im, ax=ax, label="NRMSE")

thr = (vmin + vmax) / 2.0
for j in range(nrows):
    for i in range(ncols):
        val = grid[j, i]
        ax.text(i, j, f"{val:.4f}\n±{std[j, i]:.4f}", ha="center", va="center",
                fontsize=6, color="white" if val < thr else "black")

# separate the constant reference column from the CMA-ES block
ax.axvline(len(trained_vals) - 0.5, color="white", lw=2)

ax.set_xticks(range(ncols))
ax.set_xticklabels([f"{v:g}" for v in trained_vals] + ["constant"], rotation=45, ha="right")
ax.set_yticks(range(nrows))
ax.set_yticklabels([f"{v:g}" for v in deploy_vals])
ax.set_xlabel("Trained self_connection")
ax.set_ylabel("self_connection")
ax.set_title("NRMSE Heatmap")
fig.tight_layout()
fig.savefig("/tmp/full_heatmap.png", dpi=200, bbox_inches="tight")
print(f"wrote /tmp/full_heatmap.png  ({nrows}x{ncols}, vmin={vmin:.4f} vmax={vmax:.4f})")
