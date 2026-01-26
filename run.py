from runner.grid_search import main
import sys

DEFAULT_EXP_PATH = "./experiments/weight_distribution/dir_frac/"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEFAULT_EXP_PATH = sys.argv[1]
    main(DEFAULT_EXP_PATH)