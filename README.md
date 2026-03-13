# nanomagnet-lattice-esn

Research code for running Echo State Network (ESN) experiments with lattice-based reservoirs, driven by YAML configs with optional grid search.

## What's in this repo

- **ESN core**: `ESN.py` — single-input ESN with spectral-radius scaling, Ridge readout, closed-loop evaluation, memory capacity computation.
- **Reservoir construction**: `matrix.py` — builds `W_in`, `W_res`, and optional `W_back`. Supports lattice graphs (Von Neumann/Moore), tiled weight patterns, directed/signed edges, orthogonal matrices (Haar-random, alternating projections), and loading pre-trained tiles from JSON.
- **Activation functions**: `activation.py` — `Tanh` (beta, shift, binary) and `Hysteresis` (Preisach-style per-node hysteresis with coercivity and remanence).
- **Datasets**: `data/NARMA10.py` (NARMA order 10/20/30), `data/mackey_glass.py` (Mackey-Glass with configurable tau, prediction horizon, closed-loop mode).
- **Metrics**: `metric.py` — NRMSE, kernel quality, generalization rank, memory capacity.
- **Runner**:
  - `runner/grid_search.py` — grid search, aggregation, plotting
  - `runner/single_run.py` — run a single config (NRMSE mode or reservoir metrics mode)
  - `runner/evaluation.py` — train, test, and closed-loop test functions
  - `runner/reservoir_stats.py` — reservoir state statistics (node means/variances, TRI)
- **Optimization**:
  - `optimize.py` — entry point for tile optimization
  - `optimizer/cmaes.py` — CMA-ES tile weight optimization
  - `optimizer/hyperneat.py` — HyperNEAT/CPPN tile evolution
  - `optimizer/fitness.py` — tile evaluation fitness function
- **Utilities**:
  - `utils/gs_plot.py` — grid search plotting (PNG for 1D/2D, HTML for 3D)
- **Visualization** (standalone scripts):
  - `visualization/analysis.py` — multi-panel reservoir analysis (eigenvalues, weight histograms, degree stats)
  - `visualization/eigvec_viz.py` — interactive eigenvector visualization
  - `visualization/plot_matrix.py` — `plot_tile()` and `plot_lattice()` for direct graph visualization
  - `visualization/plot_lattice_neighborhoods.py`, `visualization/plot_lattice_hysteresis.py`, `visualization/plot_tile_tiling.py` — figure scripts

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

To run tile optimization:

```bash
python optimize.py ./experiments/my_opt_exp/
```

## Config format (`config.yaml`)

A complete example template with all available fields is at `experiments/config.yaml`. Full documentation of every field is in `experiments/CONFIG_README.md`.

- **Grid search**: any YAML value that is a list becomes a grid-search dimension. All combinations are run.
- **Repetitions**: `num_runs` repeats each config combination (fresh random initialization each time).
- **Directory expansion**: if `W_res_args.type` is a directory path, it is auto-expanded into a list of all `.json` files in that directory.

Example:

```yaml
num_runs: 5
res_metrics: false        # false => train/test NRMSE; true => compute KQ/Gen/MC
state_plot: false
plot_deciles: false

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
      distribution: fixed   # fixed | uniform
    W_res_args:
      type: constant         # constant | random | tile | baseline-esn | orthogonal | path.json
      neighborhood: Moore    # Von_Neumann | Moore
      self_connection: [0.0, 0.1, 0.2]  # list => grid search
      directed_edges_fraction: 0.9
      directed_edges_weights: 0.0
      sign_frac: 0.5
```

Notes:

- **`W_args.size` must be a perfect square** when using lattice-based types (the lattice is `sqrt(size) x sqrt(size)`).
- **`data.load: true`** expects `.npy` files under `data/datasets/`. Set `load: false` to generate and save them.

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

### Optimization

- `best_tile_*.json` files containing the optimized tile graph with metadata (method, best NRMSE, parameter values).
