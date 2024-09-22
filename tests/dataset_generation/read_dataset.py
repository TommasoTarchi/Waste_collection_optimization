import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams


if __name__ == "__main__":

    # set data path
    data_dir = "./data/"

    # load parameters
    params = ProblemParams()
    params.load_from_dir(data_dir)

    print("Parameters loaded.")

    # print parameters
    print(f"num_nodes: {params.num_nodes}")
    print(f"num_edges: {params.num_edges}")
    print(f"num_required_edges: {params.num_required_edges}")
    print(f"num_vehicles: {params.num_vehicles}")
    print(f"num_periods: {params.num_periods}")
    print(f"W: {params.W}")
    print(f"T_max: {params.T_max}")
    print(f"M: {params.M}")
    print(f"theta: {params.theta}")
    print(f"sigma: {params.sigma}")
    print(f"ul: {params.ul}")
    print(f"uu: {params.uu}")

    print(f"c: {params.c}")

    for i in range(params.d.shape[2]):
        print(f"d {i + 1}:\n")
        print(params.d[:, :, i])
        print("\n")

    print(f"t: {params.t}")
    print(f"cv: {params.cv}")
    print(f"G: {params.G}")

    print(f"existing_edges: {params.existing_edges}")
    print(f"required_edges: {params.required_edges}")
