import sys
import os
import numpy as np

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset
from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
from WCO_lib.evaluate import compute_normalized_MID, compute_RASO, compute_distance


if __name__ == "__main__":

    # set output file
    output_file = "results.txt"

    # set data path
    data_dir = "./data/"

    # set bounds for dataset
    bounds_c = (1, 3)
    bounds_d = (1, 3)
    bounds_t = (1, 3)
    bounds_cv = (1, 3)
    bounds_G = (1, 3)

    # generate dataset
    generate_dataset(data_dir, bounds_c, bounds_d, bounds_t, bounds_cv, bounds_G)

    print("Dataset generated.\n")

    # load problem parameters
    problem_params = ProblemParams()
    problem_params.load_from_dir(data_dir)

    print("Parameters loaded.\n")

    # set problem
    solver_params = MosaMoiwoaSolverParams()
    solver = MosaMoiwoaSolver(problem_params, solver_params)

    print("Problem set.\n")

    solver.generate_initial_solutions()

    print("Initial solutions generated\n")

    solver.apply_MOSA()

    print("MOSA applied.\n")

    solver.apply_MOIWOA()

    # TODO: return solutions in some way

    print("Problem solved with MOIWOA.\n")

    # TODO: check solutions in some way
