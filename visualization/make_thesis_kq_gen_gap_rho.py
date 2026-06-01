import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from utils.config_loader import ConfigLoader
from runner.single_run import run_res_metrics

EXP = "experiments/v2/hysteresis/res_metrics"

np.random.seed(0)
torch.manual_seed(0)

configs, param_names = ConfigLoader.generate_grid_search_configs(EXP)
print("param_names:", param_names, "| total configs:", len(configs))

# store[type][spectral_radius] = {"kq": [...], "gen": [...]}
store = {}
for i, cfg in enumerate(configs):
    typ = cfg.conf["esn"]["W_args"]["W_res_args"]["type"]
    rho = float(cfg.conf["esn"]["spectral_radius"])
    m = run_res_metrics(cfg.conf)
    store.setdefault(typ, {}).setdefault(rho, {"kq": [], "gen": []})
    store[typ][rho]["kq"].append(float(m["kernel_quality"]))
    store[typ][rho]["gen"].append(float(m["generalization"]))
    cfg.conf.get("esn", {}).pop("model", None)
    if (i + 1) % 18 == 0:
        print(f"  {i+1}/{len(configs)} done")

TYPE_COLOR = {"baseline-esn": "C0", "constant": "C1"}
GAP_COLOR = {"baseline-esn": "green", "constant": "purple"}

plt.figure()
for typ in sorted(store.keys(), key=str):   # baseline-esn, constant
    xs = sorted(store[typ])
    x = np.array(xs)
    kq_m = np.array([np.mean(store[typ][s]["kq"]) for s in xs])
    kq_s = np.array([np.std(store[typ][s]["kq"]) for s in xs])
    gen_m = np.array([np.mean(store[typ][s]["gen"]) for s in xs])
    gen_s = np.array([np.std(store[typ][s]["gen"]) for s in xs])
    gap_runs = [np.array(store[typ][s]["kq"]) - np.array(store[typ][s]["gen"]) for s in xs]
    gap_m = np.array([g.mean() for g in gap_runs])
    gap_s = np.array([g.std() for g in gap_runs])

    c = TYPE_COLOR.get(typ, "C2")
    gc = GAP_COLOR.get(typ, "green")
    plt.plot(x, kq_m, marker="o", linestyle="-", color=c, label=f"KQ type={typ}")
    plt.fill_between(x, kq_m - kq_s, kq_m + kq_s, alpha=0.15, color=c)
    plt.plot(x, gen_m, marker="s", linestyle="--", color=c, label=f"Gen type={typ}")
    plt.fill_between(x, gen_m - gen_s, gen_m + gen_s, alpha=0.15, color=c)
    plt.plot(x, gap_m, marker="^", linestyle="-", color=gc, label=f"KQ $-$ Gen type={typ}")
    plt.fill_between(x, gap_m - gap_s, gap_m + gap_s, alpha=0.15, color=gc)

    print(f"== {typ} ==")
    for s, k, g, d in zip(xs, kq_m, gen_m, gap_m):
        print(f"  rho={s:.2f}  KQ={k:6.1f}  Gen={g:6.1f}  KQ-Gen={d:6.1f}")

plt.xlabel(r"$\rho$")
plt.ylabel("Score")
plt.title("KQ vs Gen by type")
plt.legend(fontsize="small")
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = "/tmp/spectral_radius_vs_type_kq_gen_gap_lines.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
