# Config Reference

Complete reference for `config.yaml` fields. Copy `experiments/config.yaml` as a starting template.

## Grid Search

Any YAML value that is a list automatically becomes a grid-search dimension. All combinations are generated and each is repeated `num_runs` times with a fresh random seed.

```yaml
# This runs 2 x 3 = 6 parameter combinations, each 10 times = 60 total runs
num_runs: 10
esn:
  W_args:
    W_res_args:
      neighborhood: [Von_Neumann, Moore]
      self_connection: [0.0, 0.5, 1.0]
```

**Excluded from grid search** (treated as structural, not swept):
- `esn.W_args.W_res_args.tile.shape`
- `esn.W_args.W_back_args.range`
- `optimization.cmaes.bounds`
- `optimization.hyperneat.substrate_shape`
- `optimization.hyperneat.substrate_coords`

**Nested list override**: excluded keys can still be swept by using a nested list. For example, `shape: [3, 3]` is treated as a single shape, but `shape: [[1, 1], [2, 2], [3, 3]]` is treated as a sweep over three shapes.

**Directory expansion**: if `W_res_args.type` is a directory path, it is auto-expanded into a list of all `.json` files in that directory, becoming a grid-search dimension. When `tile_replicas: true`, the files are instead treated as replicas to average (see `W_res_args.tile_replicas`).

---

## Root-Level Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `num_runs` | int | `1` | Number of repetitions per grid-search combination. Each run gets a fresh random seed. |
| `res_metrics` | bool | `false` | `false` = NRMSE mode (train ESN, evaluate prediction accuracy). `true` = reservoir metrics mode (compute kernel quality, generalization rank, and memory capacity without training on a dataset). |
| `state_plot` | bool | `false` | Save a plot of reservoir node activations over time. Only applies in NRMSE mode. |
| `plot_deciles` | bool | `false` | Plot decile bands in result aggregation. |

---

## `data` — Dataset Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | *required* | Dataset name: `"NARMA"`, `"Mackey-Glass"`, or `"Lorenz"`. |
| `load` | bool | *required* | `true` = load pre-generated `.npy` files from `data/datasets/`. `false` = generate data from scratch (and save to disk). |
| `sample_len` | int | *required* | Total number of time steps in the dataset. |
| `split_ratio` | float | *required* | Fraction of data used for training. The remainder is used for testing. Value between 0 and 1. |
| `val_ratio` | float | *optional* | Fraction of the training split held out for validation. Omit to skip validation split. |
| `system_order` | int | — | **NARMA only.** Order of the NARMA system (e.g., 10, 20, 30). |
| `tau` | int | `17` | **Mackey-Glass only.** Delay parameter of the Mackey-Glass equation. |
| `prediction_horizon` | int | `1` | **Mackey-Glass only.** Number of steps ahead to predict. |
| `closed_loop` | bool | `false` | **Mackey-Glass only.** `true` = use closed-loop free-run evaluation instead of open-loop. |
| `train_len` | int | *optional* | **Mackey-Glass only.** Explicit training set length. Overrides `split_ratio` when used together with `test_len`. |
| `test_len` | int | *optional* | **Mackey-Glass only.** Explicit test set length. Overrides `split_ratio` when used together with `train_len`. |

---

## `esn` — Echo State Network

| Field | Type | Default | Description |
|---|---|---|---|
| `spectral_radius` | float or null | *required* | Target spectral radius for `W_res`. The matrix is rescaled so its largest eigenvalue magnitude equals this value. Set to `null` to skip rescaling (use raw weights as-is). |
| `washout` | int | *required* | Number of initial timesteps discarded before training/evaluation. Lets the reservoir "warm up" to avoid transient effects. |
| `training_noise` | float | `0` | Amplitude of uniform noise added to the state update during training. Noise is sampled from `[-training_noise, +training_noise]`. Only applied when feedback connections (`W_back`) are active. |
| `input_bias` | float or null | `null` | When set to a float, replaces the actual input `u[t]` with this constant value at every timestep. Used for reservoir metrics experiments. `null` = use the real input signal. |

### `esn.f` — Activation Function

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | *required* | Activation function: `"tanh"` or `"hysteresis"`. Both names use the same `Hysteresis` class — `"tanh"` is an alias for convenience. |

#### `esn.f.args` — Activation Arguments

All parameters are shared between `"tanh"` and `"hysteresis"` names (they use the same class). With `h_c=0` and `m_r=1` the activation reduces to standard `tanh(beta * x - shift)`.

| Field | Type | Default | Description |
|---|---|---|---|
| `beta` | float | `1.0` | Steepness (gain) of the tanh. Higher values make the activation more step-like. |
| `shift` | float | `0.0` | Horizontal shift applied before tanh: `tanh(beta * x - shift)`. Moves the inflection point away from zero. |
| `binary` | bool | `false` | If `true`, applies `sign()` after tanh, producing binary {-1, +1} outputs. |
| `h_c` | float | `0.0` | Coercivity — half-width of the hysteresis loop. Controls how far the input must swing before the output switches branches. `0` = no hysteresis (standard tanh). |
| `m_r` | float | `1.0` | Remanence scaling factor. Scales the output amplitude of both branches. `1.0` = standard amplitude. |
| `decay_rate` | float | `2.0` | Minor loop merge speed. On direction reversal, the output smoothly transitions toward the target major branch; `decay_rate` controls how fast. `0` = hard branch switch (legacy behavior). Higher values = faster merge. Practical range: 0.5–5.0. |

### `esn.readout` — Readout Layer

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | *required* | Readout type: `"Ridge"`. |

#### `esn.readout.args`

| Field | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | `1.0` | Ridge regularization strength. `0` = ordinary least squares (no regularization). Larger values penalize large readout weights more strongly. |

### `esn.W_args` — Weight Matrices

| Field | Type | Default | Description |
|---|---|---|---|
| `size` | int | *required* | Number of reservoir nodes. **Must be a perfect square** when using lattice-based types (everything except `baseline-esn`), since the reservoir is arranged as a `sqrt(size) x sqrt(size)` grid. Common values: 100, 225, 400, 625. |

#### `esn.W_args.W_in_args` — Input Weights

| Field | Type | Default | Description |
|---|---|---|---|
| `input_scale` | float | *required* | Scalar multiplier applied to the input weight vector. Controls the strength of input driving the reservoir. |
| `distribution` | string | *required* | How input weights are initialized: `"uniform"` = `rand(size) - 0.5` (centered uniform), `"fixed"` = all ones (every node gets the same input). |

#### `esn.W_args.W_res_args` — Reservoir Weights

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | string | *required* | Reservoir matrix construction method. See **Reservoir Types** below. |
| `tile_replicas` | bool | `false` | When `type` is a directory path, treat all `.json` files within as replicas of the same configuration (averaged together) instead of separate grid-search dimensions. When pointing to a parent directory containing subdirectories, each subdirectory becomes a grid-search point and files within each are replicas. Use with tiles produced by `optimize.py` with `optimization.num_runs > 1`. |
| `neighborhood` | int or string | `"Von_Neumann"` | Number of Euclidean distance shells to include as neighbors. Each shell adds all grid cells at the next unique distance: `1` = 4 neighbors (distance 1, Von Neumann), `2` = 8 (+ distance √2 diagonals, Moore), `3` = 12 (+ distance 2), `4` = 20 (+ distance √5 knight's moves), etc. Also accepts legacy strings `"Von_Neumann"` (= `1`) and `"Moore"` (= `2`). Only used by lattice-based types. |
| `self_connection` | float | *required* | Weight of self-loops added to every node. `0.0` = no self-connections. |
| `sign_frac` | float | *required* | Fraction of edges whose weight is negated. `0.0` = all positive, `0.5` = half negative, `1.0` = all negative. |
| `directed_edges_fraction` | float | *required* | Fraction of bidirectional edge pairs that become unidirectional. `0.0` = fully bidirectional, `1.0` = all edges become one-way. |
| `directed_edges_weights` | float | *required* | Weight multiplier applied to the "removed" direction of a directed edge. `0.0` = fully removes it, `1.0` = keeps it (no effect). |
| `alternating_proj` | int | `0` | Number of alternating projection iterations applied to the **full W_res matrix** after construction. Each iteration: (1) SVD-project onto the orthogonal manifold, (2) re-zero all originally-zero entries. Pushes the matrix toward orthogonality while preserving sparsity. Works with all matrix types. |
| `orthogonal_tile` | bool/string | `false` | Orthogonalize the **tile** before tiling it across the lattice. `false` = off, `"exact"` = polar decomposition (tile becomes dense, exactly orthogonal), `"alternating"` = alternating projections (preserves tile sparsity). Only applies to tile-based types. |
| `orthogonal_tile_iters` | int | `50` | Number of iterations when `orthogonal_tile: alternating`. Ignored otherwise. |

##### Reservoir Types (`W_res_args.type`)

| Value | Description |
|---|---|
| `"constant"` | Lattice graph with all edge weights = 1 (before sign/direction modification). |
| `"random"` | Lattice graph with edge weights sampled uniformly from [0, 1). |
| `"tile"` | Build a small periodic tile graph, then tile it across the full lattice. Requires the `tile` sub-config. |
| `"baseline-esn"` | Traditional dense random reservoir: uniform [-1, 1] weights with 90% sparsity. Not lattice-based. |
| `"orthogonal"` | Dense Haar-random orthogonal matrix (all eigenvalues on the unit circle). No lattice structure, no post-processing. Returns immediately. |
| `"path/to/tile.json"` | Load a pre-trained tile from a JSON file and tile it onto the lattice. |
| `"path/to/directory/"` | Auto-expanded into a list of all `.json` files in the directory (grid-search over saved tiles). |

##### `"from_tile"` Value Resolution

When `type` is a path to a `.json` tile file, any field under `W_res_args` (including nested fields) can be set to the string `"from_tile"` instead of a numeric value. Before the run starts, `ConfigLoader` reads the tile JSON's `metadata.params` dictionary and replaces each `"from_tile"` field with the corresponding value stored under its dotted path (e.g., `esn.W_args.W_res_args.self_connection` or `esn.W_args.W_res_args.tile.shape`). This lets optimized tiles carry their own parameter values so experiments automatically use the settings the tile was optimized with.

```yaml
W_res_args:
  type: path/to/optimized_tile.json
  self_connection: from_tile    # resolved from tile JSON metadata
  sign_frac: from_tile          # resolved from tile JSON metadata
  directed_edges_fraction: 0.9  # kept as-is
  tile:
    shape: from_tile            # nested fields work too
```

When `type` is not a `.json` path (e.g., `"constant"`, `"baseline-esn"`), `"from_tile"` values are left as-is and ignored — those code paths don't read the tile-specific fields. This means `"from_tile"` is safe to use in configs that also sweep non-tile types.

#### `esn.W_args.W_res_args.tile` — Tile Configuration

Only used when `type: tile`.

| Field | Type | Default | Description |
|---|---|---|---|
| `shape` | [int, int] | *required* | Tile dimensions `[rows, cols]`. The tile is built as a periodic lattice of this size. **Not a grid-search dimension** (lists here are treated as the shape, not swept). |
| `method` | string | *required* | `"random"` = assign random weights to tile edges. |

#### `esn.W_args.W_back_args` — Feedback Weights (Optional)

Omit entirely or set to `null` to disable feedback connections.

| Field | Type | Default | Description |
|---|---|---|---|
| `range` | [float, float] | — | Uniform sampling range for feedback weight vector, e.g., `[-0.1, 0.1]`. **Not a grid-search dimension.** |

---

## `optimization` — Tile Optimization

Used by the optimizer scripts (`optimizer/cmaes.py`, `optimizer/hyperneat.py`) to search for optimal tile weight patterns. Not used during standard `run.py` experiments.

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | string | *required* | Optimization method: `"cmaes"` or `"hyperneat"`. |
| `num_runs` | int | `1` | Number of optimization runs per grid-search combination. Each run uses a different seed (`seed + run_index`). When > 1, tiles are saved to a subdirectory (`best_tiles_*/run_0.json`, `run_1.json`, ...). |
| `num_evals` | int | `1` | Number of ESN evaluations per candidate tile (results are averaged). Higher = more robust but slower. |
| `seed` | int | `42` | Random seed for the optimizer. |

### `optimization.cmaes` — CMA-ES Settings

| Field | Type | Default | Description |
|---|---|---|---|
| `sigma0` | float | `0.3` | Initial step size for CMA-ES. |
| `max_generations` | int | `200` | Maximum number of CMA-ES generations. |
| `pop_size` | int | `20` | Population size per generation. |
| `bounds` | [float, float] | `[-1, 1]` | Min/max bounds for tile weights. **Not a grid-search dimension.** |
| `optimize_signs` | bool | `false` | Include edge sign pattern in the optimization variables. |
| `optimize_directions` | bool | `false` | Include edge directionality in the optimization variables. |
| `optimize_self_connections` | bool | `false` | Include self-connection weights in the optimization variables. |

### `optimization.hyperneat` — HyperNEAT Settings

| Field | Type | Default | Description |
|---|---|---|---|
| `cppn_config_file` | string | *required* | Path to the NEAT/CPPN configuration file (relative to the experiment directory). |
| `generations` | int | `300` | Number of NEAT generations. |
| `threshold` | float | `0.2` | HyperNEAT connection threshold — CPPN outputs below this magnitude are set to zero. |
| `substrate` | string | `"grid"` | Substrate type: `"grid"` = regular grid, `"custom"` = user-defined coordinates. |
| `substrate_shape` | [int, int] | `[3, 3]` | Grid substrate dimensions. **Not a grid-search dimension.** |
| `substrate_coords` | list of [int, int] | — | Custom substrate node coordinates. **Not a grid-search dimension.** Only used when `substrate: custom`. |

### `optimization.output`

| Field | Type | Default | Description |
|---|---|---|---|
| `save_best_tile` | bool | `true` | Save the best tile found as a `.json` file in the experiment directory. |

---

## Outputs

All outputs are written to the experiment folder:

- **NRMSE mode** (`res_metrics: false`): `reservoir_stats_summary.json` with per-combination statistics, plus parameter sweep plots (PNG for 1D/2D, interactive HTML for 3D).
- **Metrics mode** (`res_metrics: true`): kernel quality, generalization rank, and memory capacity plots.
- **State plots** (`state_plot: true`): `state_plot.png` showing reservoir node activations over time.
- **Optimization**: `best_tile_*.json` files containing the optimized tile graph. With `optimization.num_runs > 1`, tiles are saved in subdirectories (`best_tiles_*/run_0.json`, ...).

## Filtering

Runs with NRMSE >= 0.8 are automatically filtered out from aggregation and plots (treated as failed/diverged runs).
