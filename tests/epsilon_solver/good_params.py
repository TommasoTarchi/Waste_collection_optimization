import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import compute_good_parameters


if __name__ == "__main__":

    # set data path
    data_dir = "./data/"

    # set problem size
    num_nodes = 10
    num_edges = 18
    num_required_edges = 13
    num_vehicles = 3

    # set bounds for dataset
    bounds_c = (1, 3)
    bounds_d = (1, 3)
    bounds_t = (1, 3)
    bounds_cv = (1, 3)
    bounds_G = (1, 3)

    # compute good parameters
    avg_c = (bounds_c[0] + bounds_c[1]) / 2
    avg_cv = (bounds_cv[0] + bounds_cv[1]) / 2

    good_params = compute_good_parameters(num_edges,
                                          num_required_edges,
                                          num_vehicles,
                                          3,
                                          3,
                                          avg_cv,
                                          avg_c)

    # print good parameters
    print(good_params)
