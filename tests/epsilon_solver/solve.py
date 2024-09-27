import sys
import os
import numpy as np

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset
from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver
from WCO_lib.evaluate import compute_MID, compute_RASO, compute_distance


if __name__ == "__main__":

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
    params = ProblemParams()
    params.load_from_dir(data_dir)

    print("Parameters loaded.\n")

    # set number of epsilon values for epsilon-solver
    num_epsilon = 3

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

    pareto_solutions = solver.return_pareto_solutions()

    print("Problem solved.\n")

    # print results
    print("STATUS OF THE MODELS: ", solver.return_status())
    print()

    print("PARETO SOLUTION SUMMARY:\n")
    for solution in pareto_solutions:
        print("Number of non-null x: ", np.sum(solution["x"] > 0))
        print("Number of elements x equal to one: ", np.sum(solution["x"] == 1))
        for t in range(params.num_periods):
            print(f"\tNumber of employed vehicles in period {t}: ", np.sum(solution["u"][:, t] > 0))
            print(f"\tNumber of non-null y in period {t}: ", np.sum(solution["y"][:, :, t, :] > 0))
            print(f"\tNumber of elements y equal to one in period {t}: ", np.sum(solution["y"][:, :, t, :] == 1))
            for k in range(params.num_vehicles):
                for p in range(params.num_required_edges):
                    print(f"\t\tNumber of non-null x for vehicle {k} in trip {p}: ", np.sum(solution["x"][k, p, t, :] > 0))
                    print(f"\t\tNumber of non-null y for vehicle {k} in trip {p}: ", np.sum(solution["y"][k, p, t, :] > 0))
        print()

    print("MDI FOR PARETO SOLUTIONS: ", compute_MID(params, pareto_solutions))
    print("RASO FOR PARETO SOLUTIONS: ", compute_RASO(params, pareto_solutions))
    print("DISTANCE FOR PARETO SOLUTIONS: ", compute_distance(params, pareto_solutions))
