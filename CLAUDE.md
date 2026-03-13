# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for Echo State Network (ESN) experiments with lattice-based reservoirs. Evaluates how graph topology (lattice vs random, Von Neumann vs Moore neighborhood, directed/signed edges, tiled weight patterns) affects ESN performance on time series prediction tasks (NARMA, Mackey-Glass).

## Setup & Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run an experiment (defaults to `./experiments/test/`):
```bash
python run.py
python run.py ./experiments/my_exp/
```

Run tile optimization (CMA-ES or HyperNEAT):
```bash
python optimize.py
python optimize.py ./experiments/my_opt_exp/
```

There are no tests or linters configured in this project.

## Architecture

### Entry points

- **`run.py`** → `runner/grid_search.py:main()` → `runner/single_run.py:run()` or `run_res_metrics()`
- **`optimize.py`** → runs CMA-ES or HyperNEAT tile optimization via `optimizer/` modules

### Experiment modes

Controlled by `res_metrics` in config:
- `false` (NRMSE mode): trains ESN, evaluates prediction accuracy (open-loop or closed-loop)
- `true` (metrics mode): computes kernel quality, generalization rank, memory capacity

### Grid search

Any YAML value that is a list in `config.yaml` becomes a grid-search dimension. `ConfigLoader.generate_grid_search_configs()` generates all combinations, each repeated `num_runs` times with fresh random seeds.

### Key modules

- **`ESN.py`** — `ESN(nn.Module)`: reservoir state update `x = f(W_in * u[t] + W_res @ x)`, washout, Ridge readout training, closed-loop evaluation, memory capacity computation
- **`matrix.py`** — `Matrix`: constructs `W_in`, `W_res`, optional `W_back`. Lattice mode builds a `sqrt(size) x sqrt(size)` grid graph via NetworkX. Supports Von Neumann/Moore neighborhoods, directed edges, signed weights, tiled weight patterns, orthogonal matrices (Haar-random, alternating projections), loading tiles/full matrices from JSON
- **`activation.py`** — `Tanh` (beta, shift, binary mode) and `Hysteresis` (Preisach-style per-node hysteresis with coercivity h_c and remanence m_r)
- **`readout.py`** — `Ridge`: sklearn Ridge regression wrapper
- **`metric.py`** — `nrmse`, `kernel_quality`, `generalization`, `memory_capacity`
- **`utils/config_loader.py`** — `ConfigLoader`: parses `config.yaml`, detects list params for grid search, auto-expands directory paths to `.json` file lists, instantiates all objects
- **`runner/grid_search.py`** — orchestrates grid search: runs configs, aggregates results by parameter combination, generates plots and `reservoir_stats_summary.json`
- **`runner/single_run.py`** — executes one config: `run()` for NRMSE mode, `run_res_metrics()` for reservoir metrics
- **`runner/evaluation.py`** — `train()`, `test()`, `test_closed_loop()` functions
- **`runner/reservoir_stats.py`** — computes node means/variances, mean spread, TRI, TRI ratio
- **`utils/formula.py`** — spectral radius computation, Total Recurrent Influence (TRI) via resolvent/Neumann series
- **`utils/gs_plot.py`** — 1D line plots as PNG, 2D heatmap+line plots as PNG (matplotlib), 3D scatter as interactive HTML (plotly)

### Optimization modules

- **`optimizer/cmaes.py`** — CMA-ES tile weight optimization
- **`optimizer/hyperneat.py`** — HyperNEAT/CPPN tile evolution
- **`optimizer/fitness.py`** — tile evaluation fitness function (builds Matrix via `Matrix.from_tile()`)

### Visualization (standalone scripts)

- **`visualization/analysis.py`** — multi-panel reservoir analysis (eigenvalues, weight histograms, degree stats). Run with `python -m visualization.analysis <tile.json or config.yaml>`
- **`visualization/eigvec_viz.py`** — interactive eigenvector visualization
- **`visualization/plot_matrix.py`** — `plot_tile()` and `plot_lattice()` for direct graph visualization
- **`visualization/plot_lattice_neighborhoods.py`** — Von Neumann vs Moore neighborhood illustration
- **`visualization/plot_lattice_hysteresis.py`** — lattice with self-loops and hysteresis activation
- **`visualization/plot_tile_tiling.py`** — tile tiling visualization

### Datasets

- **`data/NARMA10.py`** — NARMA system (order 10, 20, 30)
- **`data/mackey_glass.py`** — Mackey-Glass time series (configurable tau, prediction horizon, closed-loop mode)
- Lorenz is declared in ConfigLoader but not implemented

### Data flow through ConfigLoader

`ConfigLoader.__init__` chains: `_init_W()` → `_init_readout()` → `_init_f()` → `_get_data()` → `_init_esn()`. Each step instantiates objects and stores them back into the config dict. The final `conf["esn"]["model"]` is the ready-to-use ESN instance.

## Config Constraints

- `W_args.size` **must be a perfect square** when using lattice-based types (grid is `sqrt(size) x sqrt(size)`)
- `data.load: true` expects `.npy` files under `data/datasets/`; set `load: false` to generate them
- Runs with NRMSE >= 0.8 are filtered out from aggregation/plots
- Grid search excludes these list-valued fields (treated as structural): `tile.shape`, `W_back_args.range`, `cmaes.bounds`, `hyperneat.substrate_shape`, `hyperneat.substrate_coords`

## Config Documentation

When adding new config fields or changing existing ones, always update both:
- `experiments/config.yaml` — the example template with all available fields
- `experiments/CONFIG_README.md` — the full reference documenting every config field

## Outputs

All outputs are written to the experiment folder. NRMSE mode produces `reservoir_stats_summary.json` and parameter sweep plots. Metrics mode produces kernel quality, generalization, and memory capacity plots. Optimization produces `best_tile_*.json` files.
