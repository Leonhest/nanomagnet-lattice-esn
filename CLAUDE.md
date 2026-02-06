# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for Echo State Network (ESN) experiments with lattice-based reservoirs. Evaluates how graph topology (lattice vs random, Von Neumann vs Moore neighborhood, directed/signed edges, tiled weight patterns) affects ESN performance on NARMA time series prediction tasks.

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

There are no tests or linters configured in this project.

## Architecture

**Execution flow:** `run.py` → `runner/grid_search.py:main()` → `runner/single_run.py:run()` or `run_res_metrics()`

**Two experiment modes** controlled by `res_metrics` in config:
- `false` (NRMSE mode): trains ESN on NARMA data, evaluates prediction accuracy
- `true` (metrics mode): computes kernel quality, generalization, memory capacity (no dataset needed)

**Grid search**: any YAML value that is a list in `config.yaml` becomes a grid-search dimension. `ConfigLoader.generate_grid_search_configs()` generates all combinations, each repeated `num_runs` times with fresh random seeds.

### Key modules

- **`ESN.py`** — `ESN(nn.Module)`: forward pass evolves reservoir state as `x = f(W_in * u[t] + W_res @ x)`, applies washout, trains Ridge readout
- **`matrix.py`** — `Matrix`: constructs `W_in` (input weights) and `W_res` (recurrent weights). Lattice mode builds a `sqrt(size) x sqrt(size)` grid graph via NetworkX; supports Von Neumann/Moore neighborhoods, directed edges, signed weights, tiled weight patterns
- **`utils/config_loader.py`** — `ConfigLoader`: parses `config.yaml`, detects list params for grid search, instantiates all objects (Matrix → ESN → dataset)
- **`runner/grid_search.py`** — orchestrates grid search: runs configs, aggregates results by parameter combination, generates plots and `reservoir_stats_summary.json`
- **`runner/single_run.py`** — executes one config: `run()` for NRMSE mode, `run_res_metrics()` for reservoir metrics
- **`utils/formula.py`** — spectral radius computation, Total Recurrent Influence (TRI) via Neumann series approximation
- **`utils/gs_plot.py`** — 1D plots as PNG (matplotlib), ≥2D as interactive HTML (plotly)

### Data flow through ConfigLoader

`ConfigLoader.__init__` chains: `_init_W()` → `_init_readout()` → `_init_f()` → `_get_data()` → `_init_esn()`. Each step instantiates objects and stores them back into the config dict. The final `conf["esn"]["model"]` is the ready-to-use ESN instance.

## Config Constraints

- `W_args.size` **must be a perfect square** when `lattice: true` (grid is `sqrt(size) x sqrt(size)`)
- `data.load: true` expects `.npy` files under `data/datasets/NARMA10/`; set `load: false` to generate them
- Runs with NRMSE >= 0.8 are filtered out from aggregation/plots
- `plot_deciles: true` only works with ≤2 grid search parameters

## Outputs

All outputs are written to the experiment folder. NRMSE mode produces `reservoir_stats_summary.json` and parameter sweep plots. Metrics mode produces kernel quality, generalization, and memory capacity plots.
