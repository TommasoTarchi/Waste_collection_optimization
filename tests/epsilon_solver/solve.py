import sys
import os
import numpy as np

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset
from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver
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

    # compute epsilon values
    solver.compute_epsilon(num_epsilon)

    print("Epsilon values for epsilon-solver computed.\n")

    # solve multi-objective problem
    solver.solve_multi_objective()

    pareto_solutions = solver.return_pareto_solutions()

    print("Problem solved.\n")

    # print results
    with open(output_file, "w") as f:
        f.write("PARAMETERS:\n")
        f.write("Number of nodes: " + str(params.num_nodes) + "\n")
        f.write("Number of edges: " + str(params.num_edges) + "\n")
        f.write("Number of required edges: " + str(params.num_required_edges) + "\n")
        f.write("Number of periods: " + str(params.num_periods) + "\n")
        f.write("Number of vehicles: " + str(params.num_vehicles) + "\n")
        f.write("W: " + str(params.W) + "\n")
        f.write("T_max: " + str(params.T_max) + "\n")
        f.write("M: " + str(params.M) + "\n")
        f.write("theta: " + str(params.theta) + "\n")
        f.write("sigma: " + str(params.sigma) + "\n")
        f.write("ul: " + str(params.ul) + "\n")
        f.write("uu: " + str(params.uu) + "\n")

        f.write("\nALL EDGES: " + str(params.existing_edges) + "\n")
        f.write("REQUIRED EDGES: " + str(params.required_edges) + "\n")

        f.write("\nSTATUS OF THE MODELS: " + str(solver.return_status()) + "\n")

        f.write("\nPARETO SOLUTION SUMMARY:\n")
        for solution in pareto_solutions:
            f.write("Number of non-null x: " + str(np.sum(solution["x"] > 0)) + "\n")
            f.write("Number of elements x equal to one: " + str(np.sum(solution["x"] == 1)) + "\n")
            for t in range(params.num_periods):
                f.write("\tNumber of employed vehicles in period " + str(t) + ": " + str(np.sum(solution["u"][:, t] > 0)) + "\n")
                f.write("\tNumber of non-null y in period " + str(t) + ": " + str(np.sum(solution["y"][:, :, t, :] > 0)) + "\n")
                f.write("\tNumber of elements y equal to one in period " + str(t) + ": " + str(np.sum(solution["y"][:, :, t, :] == 1)) + "\n")
                for k in range(params.num_vehicles):
                    f.write("\t\tu: " + str(solution["u"][k, t]) + "\n")
                    for p in range(params.num_required_edges):
                        f.write("\t\tNumber of non-null x for vehicle " + str(k) + " in trip " + str(p) + ": " + str(np.sum(solution["x"][k, p, t, :] > 0)) + "\n")
                        f.write("\t\tNumber of non-null y for vehicle " + str(k) + " in trip " + str(p) + ": " + str(np.sum(solution["y"][k, p, t, :] > 0)) + "\n")
                        f.write("\t\tx: " + str(solution["x"][k, p, t, :]) + "\n")
                        f.write("\t\ty: " + str(solution["y"][k, p, t, :]) + "\n")

        f.write("\nNORMALIZED MID FOR PARETO SOLUTIONS: " + str(compute_normalized_MID(params, pareto_solutions)) + "\n")
        f.write("\nRASO FOR PARETO SOLUTIONS: " + str(compute_RASO(params, pareto_solutions)) + "\n")
        f.write("\nDISTANCE FOR PARETO SOLUTIONS: " + str(compute_distance(params, pareto_solutions)) + "\n")
