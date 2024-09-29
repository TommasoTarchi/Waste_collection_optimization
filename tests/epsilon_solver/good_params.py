import sys
import os
import json

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import compute_good_parameters


if __name__ == "__main__":

    # read problem size from json
    with open('./data/problem_parameters.json') as f:
        data = json.load(f)
        num_edges = data['num_edges']
        num_required_edges = data['num_required_edges']
        num_vehicles = data['num_vehicles']

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
