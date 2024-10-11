import sys
import os
import json
import argparse

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import compute_good_parameters


if __name__ == "__main__":

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--num_nodes', type=int, default=None)
    parser.add_argument('--num_edges', type=int, default=None)
    parser.add_argument('--num_required_edges', type=int, default=None)
    parser.add_argument('--num_vehicles', type=int, default=None)
    parser.add_argument('--num_periods', type=int, default=None)

    args = parser.parse_args()

    data_dir = args.data_dir
    num_nodes = args.num_nodes
    num_edges = args.num_edges
    num_required_edges = args.num_required_edges
    num_vehicles = args.num_vehicles
    num_periods = args.num_periods

    # compute good parameters (assuming all intervals are [1, 3])
    good_params = compute_good_parameters(num_edges,
                                          num_required_edges,
                                          num_vehicles,
                                          3,
                                          3,
                                          2,
                                          2)

    # write good parameters to json
    new_params = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_required_edges": num_required_edges,
        "num_periods": num_periods,
        "num_vehicles": num_vehicles,
        "W": good_params["W"],
        "T_max": good_params["T_max"],
        "M": 10000,
        "theta": good_params["theta"],
        "sigma": 10,
        "ul": good_params["ul"],
        "uu": good_params["uu"]
    }

    with open(os.path.join(data_dir, "problem_parameters.json"), 'w') as f:
        json.dump(new_params, f, indent=2)

    print("Good parameters written to", os.path.join(data_dir, "problem_parameters.json"))
