import sys
import os
import argparse

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)

    args = parser.parse_args()

    data_dir = args.data_dir

    # set bounds for dataset
    bounds_c = (1, 3)
    bounds_d = (1, 3)
    bounds_t = (1, 3)
    bounds_cv = (1, 3)
    bounds_G = (1, 3)

    # generate dataset
    generate_dataset(data_dir,
                     bounds_c,
                     bounds_d,
                     bounds_t,
                     bounds_cv,
                     bounds_G)

    print(f"Dataset generated in {data_dir}")
