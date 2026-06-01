import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from utils.config_loader import ConfigLoader
from runner.single_run import run_res_metrics
from runner.grid_search import _extract_param_key
from utils.gs_plot import _encode_column

EXP = "experiments/v2/tiles/normal/res_metrics"

np.random.seed(0)
torch.manual_seed(0)

configs, pnames = ConfigLoader.generate_grid_search_configs(EXP)
print("param_names:", pnames, "| total configs:", len(configs))

store = {}        # typeval -> {"kq": [...], "gen": [...]}
order_seen = []   # preserve first-seen order of category values
for i, cfg in enumerate(configs):
    key = _extract_param_key(cfg.conf, pnames)[0]
    m = run_res_metrics(cfg.conf)
    if key not in store:
        store[key] = {"kq": [], "gen": []}
        order_seen.append(key)
    store[key]["kq"].append(float(m["kernel_quality"]))
    store[key]["gen"].append(float(m["generalization"]))
    cfg.conf.get("esn", {}).pop("model", None)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(configs)} done")

# encode categories exactly like the pipeline (natural sort + display labels)
cats = list(store.keys())
x_enc, ticks = _encode_column(cats)            # x_enc[i] <-> cats[i]
kq_m = np.array([np.mean(store[c]["kq"]) for c in cats])
kq_s = np.array([np.std(store[c]["kq"]) for c in cats])
gen_m = np.array([np.mean(store[c]["gen"]) for c in cats])
gen_s = np.array([np.std(store[c]["gen"]) for c in cats])
gap_runs = [np.array(store[c]["kq"]) - np.array(store[c]["gen"]) for c in cats]
gap_m = np.array([g.mean() for g in gap_runs])
gap_s = np.array([g.std() for g in gap_runs])

order = np.argsort(x_enc)
x = x_enc[order]
kq_m, kq_s = kq_m[order], kq_s[order]
gen_m, gen_s = gen_m[order], gen_s[order]
gap_m, gap_s = gap_m[order], gap_s[order]
labels = ticks if ticks is not None else [str(c) for c in np.array(cats)[order]]

for lab, k, g, d in zip(labels, kq_m, gen_m, gap_m):
    print(f"  {lab:12s}  KQ={k:6.1f}  Gen={g:6.1f}  KQ-Gen={d:6.1f}")

plt.figure()
plt.plot(x, kq_m, marker="o", label="Kernel Quality", linewidth=2)
plt.plot(x, gen_m, marker="s", label="Generalization", linewidth=2)
plt.plot(x, gap_m, marker="^", color="green", label="KQ $-$ Gen", linewidth=2)
plt.fill_between(x, kq_m - kq_s, kq_m + kq_s, alpha=0.2)
plt.fill_between(x, gen_m - gen_s, gen_m + gen_s, alpha=0.2)
plt.fill_between(x, gap_m - gap_s, gap_m + gap_s, alpha=0.2, color="green")
if ticks is not None:
    plt.xticks(range(len(ticks)), ticks)
plt.xlabel("type")
plt.ylabel("Score")
plt.title("Kernel Quality vs Generalization")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = "/tmp/type_kq_gen_gap.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
