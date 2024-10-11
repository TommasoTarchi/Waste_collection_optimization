import sys
import os
import numpy as np
import argparse
import time

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver
from WCO_lib.evaluate import compute_normalized_MID, compute_RASO, compute_distance


if __name__ == "__main__":

    # get number of epsilon values
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epsilon", type=int, default=4)

    args = parser.parse_args()

    num_epsilon = args.num_epsilon

    # set output file
    output_file = "./results/test/results_epsilon.txt"

    # set data path
    data_dir = "./data/test/"

    # load problem parameters
    params = ProblemParams()
    params.load_from_dir(data_dir)

    print("Parameters loaded.\n")

    # set solver for problem
    solver = EpsilonSolver(params)

    print("Problem set.\n")

    t0 = time.perf_counter()

    # solve single-objective problems
    solver.solve_single_objectives()

    t1 = time.perf_counter()

    # compute epsilon values
    solver.compute_epsilon(num_epsilon)

    t2 = time.perf_counter()

    print("Epsilon values for epsilon-solver computed.\n")

    t3 = time.perf_counter()

    # solve multi-objective problem
    solver.solve_multi_objective()

    t4 = time.perf_counter()

    pareto_solutions = solver.return_pareto_solutions()

    print("Problem solved.\n")

    # compute profiling
    time_single_obj = t1 - t0
    time_comp_epsilon = t2 - t1
    time_multi_obj = t4 - t3

    # print results summary
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

        f.write("\nPROFILING:\n")
        f.write("Time for single-objective problems resolution: " + str(time_single_obj) + " seconds\n")
        f.write("Time for epsilon values computation: " + str(time_comp_epsilon) + " seconds\n")
        f.write("Time for final model resolution: " + str(time_multi_obj) + " seconds\n")
        f.write("Total time for rsolution: " + str(time_single_obj + time_comp_epsilon + time_multi_obj) + " seconds\n")

        f.write("\nEDGES LIST: " + str(params.existing_edges) + "\n")
        f.write("REQUIRED EDGES LIST: " + str(params.required_edges) + "\n")

        f.write("\nGUROBI STATUS OF THE SOLVED MODELS: " + str(solver.return_status()) + "\n")

        f.write("\nEVALUATION METRICS:\n")
        f.write("Normalized mid for pareto solutions: " + str(compute_normalized_MID(params, pareto_solutions)) + "\n")
        f.write("Raso for pareto solutions: " + str(compute_RASO(params, pareto_solutions)) + "\n")
        f.write("Distance for pareto solutions: " + str(compute_distance(params, pareto_solutions)) + "\n")

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
