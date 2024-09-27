import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import compute_good_parameters_random


if __name__ == "__main__":

    # set data path
    data_dir = "./data/"

    # set problem size
    num_nodes = 6
    num_edges = 10
    num_required_edges = 4
    num_vehicles = 2

    # set bounds for dataset
    bounds_c = (1, 6)
    bounds_d = (1, 3)
    bounds_t = (1, 5)
    bounds_cv = (1, 10)
    bounds_G = (1, 4)

    # compute good parameters
    avg_c = (bounds_c[0] + bounds_c[1]) / 2
    avg_cv = (bounds_cv[0] + bounds_cv[1]) / 2

    good_params = compute_good_parameters_random(num_edges,
                                                 num_required_edges,
                                                 num_vehicles,
                                                 3,
                                                 5,
                                                 avg_cv,
                                                 avg_c)

    # print good parameters
    print(good_params)
