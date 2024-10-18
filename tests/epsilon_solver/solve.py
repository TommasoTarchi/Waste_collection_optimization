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

    # set number of epsilon values for epsilon-solver
    num_epsilon = 4

    # generate dataset
    generate_dataset(data_dir, bounds_c, bounds_d, bounds_t, bounds_cv, bounds_G)

    print("Dataset generated.\n")

    # load problem parameters
    params = ProblemParams()
    params.load_from_dir(data_dir)

    print("Parameters loaded.\n")

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
        f.write("PROBLEM PARAMETERS:\n")
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

        f.write("\nEDGES LIST: " + str(params.existing_edges) + "\n")
        f.write("REQUIRED EDGES LIST: " + str(params.required_edges) + "\n")

        f.write("\nGUROBI STATUS OF THE SOLVED MODELS: " + str(solver.return_status()) + "\n")

        f.write("\nEVALUATION METRICS:\n")
        f.write("Number of Pareto solutions: " + str(len(pareto_solutions)))
        f.write("Normalized MID for pareto solutions: " + str(compute_normalized_MID(params,
                                                                                     solutions=pareto_solutions)) + "\n")
        f.write("RASO for pareto solutions: " + str(compute_RASO(params, solutions=pareto_solutions)) + "\n")
        f.write("Distance for pareto solutions: " + str(compute_distance(params, solutions=pareto_solutions)) + "\n")

        f.write("\nPARETO SOLUTIONS SUMMARY:\n")
        solution_count = 0
        for solution in pareto_solutions:
            f.write("Solution " + str(solution_count) + ":\n")
            for t in range(params.num_periods):
                f.write("\tNumber of employed vehicles in period " + str(t) + ": " + str(np.sum(solution["u"][:, t] > 0)) + "\n")
                f.write("\tTotal number of traversings in period " + str(t) + ": " + str(np.sum(solution["x"][:, :, t, :] > 0)) + "\n")
                f.write("\tTotal number of served edges in period " + str(t) + ": " + str(np.sum(solution["y"][:, :, t, :] > 0)) + "\n")
                for k in range(params.num_vehicles):
                    f.write("\tVehicle " + str(k) + ":\n")
                    for p in range(params.num_required_edges):
                        f.write("\t\tTrip " + str(p) + ":\n")
                        f.write("\t\tx: " + str(solution["x"][k, p, t, :]) + "\n")
                        f.write("\t\ty: " + str(solution["y"][k, p, t, :]) + "\n")
            solution_count += 1
