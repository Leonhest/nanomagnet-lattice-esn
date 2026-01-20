from runner.grid_search import main
import sys

DEFAULT_EXP_PATH = "./experiments/neigborhood/"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        DEFAULT_EXP_PATH = sys.argv[1]
    main(DEFAULT_EXP_PATH)