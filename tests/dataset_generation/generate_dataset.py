import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset


if __name__ == "__main__":

    # set data path
    data_dir = "./data/"

    # set bounds for dataset
    bounds_c = (1, 6)
    bounds_d = (1, 3)
    bounds_t = (1, 5)
    bounds_cv = (1, 10)
    bounds_G = (1, 4)
    T_max = 100

    # generate dataset
    generate_dataset(data_dir, bounds_c, bounds_d, bounds_t, bounds_cv, bounds_G)

    print("Dataset generated.")
