#
# Run MOSA-MOIWOA on a test problem.
#
# The parameters to be used can be set by the user through a JSON.
#


import sys
import os
import argparse
import time

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
from WCO_lib.evaluate import compute_MID, compute_distance


if __name__ == "__main__":

    # get solver parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver_params_file",
                        type=str,
                        default=None,
                        required=False)

    args = parser.parse_args()

    solver_params = MosaMoiwoaSolverParams()

    if args.solver_params_file is None:
        print("MOSA-MOIWOA parameters file not provided - using default parameters.")
    else:
        print(f"Using MOSA-MOIWOA parameters from file: {args.solver_params_file}")
        solver_params.load_from_file(args.solver_params_file)

    # set output file
    output_file = "./results/test/results.txt"

    # set data path
    data_dir = "../datasets/test/"

    # load problem parameters and build graph
    problem_params = ProblemParams()
    problem_params.load_from_dir(data_dir)
    problem_params.build_graph()

    print("Problem parameters loaded.")

    # set solver for problem
    solver = MosaMoiwoaSolver(problem_params, solver_params)

    print("Problem set.")

    # generate initial solutions with heuristic
    t0 = time.perf_counter()
    solver.generate_initial_solutions()
    t1 = time.perf_counter()

    print("Initial solutions generated.")

    # modify initial solutions with MOSA
    t2 = time.perf_counter()
    solver.apply_MOSA()
    t3 = time.perf_counter()

    print("Initial solutions refined with MOSA.")
    print("Running MOIWOA...", end=" ", flush=True)

    # run MOIWOA
    t4 = time.perf_counter()
    solver.apply_MOIWOA()
    t5 = time.perf_counter()

    pareto_solutions = solver.return_pareto_solutions(stage="final")
    objective_values = solver.return_objectives(stage="final")

    print("MOIWOA run successfully.")

    # compute profiling
    time_initial_heuristic = t1 - t0
    time_MOSA = t3 - t2
    time_MOIWOA = t5 - t4

    # print results summary
    with open(output_file, "w") as f:
        f.write("PROBLEM PARAMETERS:\n")
        f.write("Number of nodes: " + str(problem_params.num_nodes) + "\n")
        f.write("Number of edges: " + str(problem_params.num_edges) + "\n")
        f.write("Number of required edges: " + str(problem_params.num_required_edges) + "\n")
        f.write("Number of periods: " + str(problem_params.num_periods) + "\n")
        f.write("Number of vehicles: " + str(problem_params.num_vehicles) + "\n")
        f.write("W: " + str(problem_params.W) + "\n")
        f.write("T_max: " + str(problem_params.T_max) + "\n")
        f.write("M: " + str(problem_params.M) + "\n")
        f.write("theta: " + str(problem_params.theta) + "\n")
        f.write("sigma: " + str(problem_params.sigma) + "\n")
        f.write("ul: " + str(problem_params.ul) + "\n")
        f.write("uu: " + str(problem_params.uu) + "\n")

        f.write("\nPROFILING:\n")
        f.write("Time for initial solutions generation: " + str(time_initial_heuristic) + " seconds\n")
        f.write("Time for MOSA: " + str(time_MOSA) + " seconds\n")
        f.write("Time for MOIWOA: " + str(time_MOIWOA) + " seconds\n")
        f.write("Total time for resolution: " + str(time_initial_heuristic + time_MOSA + time_MOIWOA) + " seconds\n")

        f.write("\nEDGES LIST: " + str(problem_params.existing_edges) + "\n")
        f.write("REQUIRED EDGES LIST: " + str(problem_params.required_edges) + "\n")

        f.write("\nEVALUATION METRICS:\n")
        f.write("Number of Pareto solutions: " + str(len(pareto_solutions)) + "\n")
        f.write("MID for Pareto solutions: " + str(compute_MID(problem_params,
                                                                                     objectives=objective_values)) + "\n")
        f.write("Distance for Pareto solutions: " + str(compute_distance(problem_params, objectives=objective_values)) + "\n")

        f.write("\nPARETO SOLUTIONS:\n")
        solution_count = 0
        for solution in pareto_solutions:
            f.write("Solution " + str(solution_count) + ":\n")
            for period in range(problem_params.num_periods):
                f.write(f"\tPeriod {period}:\n")
                f.write("\tVisited edges:" + str(solution[period]["first_part"]) + "\n")
                f.write("\tVehicles employed:" + str(solution[period]["second_part"]) + "\n")
            solution_count += 1
