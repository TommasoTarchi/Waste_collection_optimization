#
# Measure Pareto front exploration for increasing number of epsilon values.
#

import sys
import os
import csv
import time

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver
from WCO_lib.evaluate import compute_MID, compute_distance


if __name__ == "__main__":

    # set epsilon values for epsilon-constraint solver
    num_epsilon_list = [5, 6, 7, 8, 9, 10, 11, 12]

    # set data path
    data_dir = "../datasets/scalability/2"

    # set output files
    output_file = "./results/results.csv"

    # load problem parameters and build graph
    problem_params = ProblemParams()
    problem_params.load_from_dir(data_dir)

    print("Problem parameters loaded.")

    # initialize lists for results
    results = [["num_epsilon", "total_time", "NOS", "MID", "distance"]]

    for num_epsilon in num_epsilon_list:

        print(f"Run epsilon-constraint method with number of epsilon values = {num_epsilon}...", end=" ", flush=True)

        # set epsilon-constraint solver for problem
        solver = EpsilonSolver(problem_params)

        t_start = time.perf_counter()

        # solve single-objective problems
        solver.solve_single_objectives()

        # compute epsilon values
        solver.compute_epsilon(num_epsilon)

        # solve multi-objective problem
        solver.solve_multi_objective()

        total_time = time.perf_counter() - t_start

        # compute metrics
        pareto_solutions = solver.return_pareto_solutions()
        NOS = len(pareto_solutions)
        MID = compute_MID(problem_params, solutions=pareto_solutions)
        distance = compute_distance(problem_params, solutions=pareto_solutions)

        results.append([num_epsilon, total_time, NOS, MID, distance])

        print("Done.")

    # write results to csv files
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
