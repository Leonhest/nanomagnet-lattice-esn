"""
Thin entrypoint.

The original `run.py` grew a lot of logic (evaluation, stats, single-run, grid
search orchestration). That logic now lives under `runner/` so it can be reused
and kept maintainable. This file stays as the user-facing entrypoint.
"""

from runner.evaluation import test, train
from runner.grid_search import main
from runner.reservoir_stats import compute_reservoir_statistics as _compute_reservoir_statistics
from runner.single_run import run, run_res_metrics

# Keep the old default for convenience (edit this path when switching experiments)
DEFAULT_EXP_PATH = "./experiments/test/"


if __name__ == "__main__":
    main(DEFAULT_EXP_PATH)