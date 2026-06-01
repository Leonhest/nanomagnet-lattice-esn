import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from utils.config_loader import ConfigLoader
from runner.single_run import run_res_metrics

EXP = "experiments/v2/neighborhood/res_metrics"
PARAM = "neighborhood"   # esn.W_args.W_res_args.neighborhood
XLABEL = "neighborhood"

np.random.seed(0)
torch.manual_seed(0)

configs, param_names = ConfigLoader.generate_grid_search_configs(EXP)
print("param_names:", param_names, "| total configs:", len(configs))
assert len(param_names) == 1, param_names

store = {}
for i, cfg in enumerate(configs):
    pv = cfg.conf["esn"]["W_args"]["W_res_args"][PARAM]
    pv = float(pv)
    m = run_res_metrics(cfg.conf)
    store.setdefault(pv, {"kq": [], "gen": []})
    store[pv]["kq"].append(float(m["kernel_quality"]))
    store[pv]["gen"].append(float(m["generalization"]))
    cfg.conf.get("esn", {}).pop("model", None)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(configs)} done")

xs = sorted(store)
kq_m = np.array([np.mean(store[s]["kq"]) for s in xs])
kq_s = np.array([np.std(store[s]["kq"]) for s in xs])
gen_m = np.array([np.mean(store[s]["gen"]) for s in xs])
gen_s = np.array([np.std(store[s]["gen"]) for s in xs])
gap_runs = [np.array(store[s]["kq"]) - np.array(store[s]["gen"]) for s in xs]
gap_m = np.array([g.mean() for g in gap_runs])
gap_s = np.array([g.std() for g in gap_runs])
x = np.array(xs)

for s, k, g, d in zip(xs, kq_m, gen_m, gap_m):
    print(f"  {XLABEL}={s:g}  KQ={k:6.1f}  Gen={g:6.1f}  KQ-Gen={d:6.1f}")

plt.figure()
plt.plot(x, kq_m, marker="o", label="Kernel Quality", linewidth=2)
plt.plot(x, gen_m, marker="s", label="Generalization", linewidth=2)
plt.plot(x, gap_m, marker="^", color="green", label="KQ $-$ Gen", linewidth=2)
plt.fill_between(x, kq_m - kq_s, kq_m + kq_s, alpha=0.2)
plt.fill_between(x, gen_m - gen_s, gen_m + gen_s, alpha=0.2)
plt.fill_between(x, gap_m - gap_s, gap_m + gap_s, alpha=0.2, color="green")
plt.xlabel(XLABEL)
plt.ylabel("Score")
plt.title("Kernel Quality vs Generalization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = "/tmp/neighborhood_kq_gen_gap.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
