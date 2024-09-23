import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset
from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver


if __name__ == "__main__":

    # set data path
    data_dir = "./data/"

    # set bounds for dataset
    bounds_c = (1, 6)
    bounds_d = (1, 3)
    bounds_t = (1, 5)
    bounds_cv = (1, 10)
    bounds_G = (1, 4)

    # generate dataset
    generate_dataset(data_dir, bounds_c, bounds_d, bounds_t, bounds_cv, bounds_G)

    print("Dataset generated.\n")

    # load problem parameters
    params = ProblemParams()
    params.load_from_dir(data_dir)

    print("Parameters loaded.\n")

    # set number of epsilon values for epsilon-solver
    num_epsilon = 4

    # set problem
    solver = EpsilonSolver(params)

    print("Problem set.\n")

    # solve single-objective problems
    solver.solve_single_objectives()

    print("Single-objective problems solved.\n")

    # compute epsilon values
    solver.compute_epsilon(num_epsilon)

    print("Epsilon values for epsilon-solver computed.\n")

    # solve multi-objective problem
    solver.solve_multi_objective()

    print("Problem solved.\n")
