# nanomagnet-lattice-esn

Research code for running Echo State Network (ESN) experiments with lattice/directed/signed reservoirs, driven by YAML configs with optional grid search.

## What’s in this repo

- **ESN core**: `ESN.py` — single-input ESN, spectral-radius scaling, Ridge readout, optional state plotting.
- **Reservoir construction**: `matrix.py` — builds `W_in` and `W_res` (either random sparse or a lattice graph via NetworkX).
- **Datasets**: `data/NARMA10.py` — load or generate NARMA time series.
- **Metrics**: `metric.py` — NRMSE plus reservoir metrics: kernel quality, generalization, memory capacity.
- **Runner**:
  - `runner/grid_search.py` — grid search, aggregation, plotting
  - `runner/single_run.py` — run a single config (NRMSE mode or res-metrics mode)
  - `runner/reservoir_stats.py` — reservoir state statistics
  - `utils/gs_plot.py` — plotting utilities (PNG for 1D, HTML for ≥2D)

## Setup

Dependencies are pinned in `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running an experiment

1. Create an experiment folder containing a `config.yaml`, e.g. `experiments/my_exp/config.yaml`
2. Run the entrypoint:

```bash
python run.py
```

`run.py` uses `DEFAULT_EXP_PATH = "./experiments/test/"`. To run another folder without editing code:

```bash
python run.py ./experiments/my_exp/
```

## Config format (`config.yaml`)

- **Grid search**: any YAML value that is a list becomes a grid-search dimension. All combinations are run.
- **Repetitions**: `num_runs` repeats each config combination (fresh random initialization each time).

Example:

```yaml
num_runs: 5
res_metrics: false        # false => train/test NRMSE on dataset; true => compute KQ/Gen/MC
state_plot: false         
plot_deciles: false       # if true, compute per-decile stats (<=2 grid params for plotting)

data:
  name: NARMA
  load: true
  system_order: 10
  sample_len: 5000
  split_ratio: 0.25

esn:
  spectral_radius: 0.9
  washout: 200
  f:
    name: tanh
    args:
      beta: 1.0
      shift: 0.15
  readout:
    name: Ridge
    args:
      alpha: 0.0
  W_args:
    size: 400
    W_in_args:
      input_scale: 0.1
      distribution: fixed   # fixed|uniform
    W_res_args:
      lattice: true
      self_connection: [0.0, 0.1, 0.2]  # list => grid search
      directed: 0.9
      sign_frac: 0.5
```

Notes:

- **`W_args.size` must be a perfect square when `lattice: true`** (the lattice is constructed as `sqrt(size) x sqrt(size)`).
- **`data.load: true`** expects `.npy` files under `data/datasets/NARMA10/`. Set `load: false` to generate and save them.

## Outputs

All outputs are written into the experiment folder you run (the `exp_path` you pass to `main(...)`).

### NRMSE mode (`res_metrics: false`)

- **`reservoir_stats_summary.json`**: per-config aggregated stats + averaged NRMSE (`score`) and `score_std`
- **Plots**
  - **1 grid parameter**: `plot_<param>.png`
  - **≥2 grid parameters**: `plot_<param1>_vs_<param2>[...].html`
  - **Extra stat plots** (examples):
    - `plot_<param>_avg_node_mean.png`
    - `plot_<param>_avg_node_variance.png`
    - `plot_<param>_mean_spread.png`
    - `plot_<param>_avg_total_recurrent_influence.png`
- **Filtering**: runs with `NRMSE >= 0.8` are excluded from aggregation/plots.

### Reservoir-metrics mode (`res_metrics: true`)

- **Plots** (based on whatever scalar fields are present in the aggregated summary)
  - `plot_<param>_memory_capacity.png`
  - `plot_<param>_kq_gen.png` (1D only; Kernel Quality + Generalization on the same figure)
  - `plot_<param>_kernel_quality.png` and `plot_<param>_generalization.png` are skipped in 1D when the combined plot is produced.

### Deciles (`plot_deciles: true`)

- Computes per-decile statistics over reservoir states and writes HTML plots via `utils/gs_plot.py:plot_decile_statistics`.

