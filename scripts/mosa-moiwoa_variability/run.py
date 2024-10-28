#
# Measure performance variation of MOSA-MOIWOA.
#

import sys
import os
import csv
import argparse
import time

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
from WCO_lib.evaluate import compute_MID, compute_distance


if __name__ == "__main__":

    # get parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=20)
    parser.add_argument("--solver_params_file",
                        type=str,
                        default=None,
                        required=False)

    args = parser.parse_args()

    num_runs = args.num_runs

    params = MosaMoiwoaSolverParams()

    if args.solver_params_file is None:
        print("MOSA-MOIWOA parameters file not provided - using default parameters.")
    else:
        print(f"Using MOSA-MOIWOA parameters from file: {args.solver_params_file}")
        params.load_from_file(args.solver_params_file)

    # set data path
    data_dir = "../datasets/2"

    # set output files
    output_time = "./results/time.csv"
    output_NOS = "./results/NOS.csv"
    output_MID = "./results/MID.csv"
    output_distance = "./results/distance.csv"

    # load problem parameters and build graph
    problem_params = ProblemParams()
    problem_params.load_from_dir(data_dir)
    problem_params.build_graph()

    print("Problem parameters loaded.")

    # initialize lists for metrics
    time_results = [["run_id", "time_initial_heuristic", "time_MOSA", "time_MOIWOA", "total_time"]]
    NOS_results = [["run_id", "NOS"]]
    MID_results = [["run_id", "MID"]]
    distance_results = [["run_id", "distance"]]

    for run_id in range(num_runs):

        print(f"Run MOSA-MOIWOA {run_id + 1} of {num_runs}...", end=" ", flush=True)

        # set MOSA-MOIWOA solver for problem
        solver = MosaMoiwoaSolver(problem_params, params)

        # generate initial solutions with heuristic
        t0 = time.perf_counter()
        solver.generate_initial_solutions()
        t1 = time.perf_counter()

        # modify initial solutions with MOSA
        t2 = time.perf_counter()
        solver.apply_MOSA()
        t3 = time.perf_counter()

        # run MOIWOA
        t4 = time.perf_counter()
        solver.apply_MOIWOA()
        t5 = time.perf_counter()

        # compute profiling
        time_initial_heuristic = t1 - t0
        time_MOSA = t3 - t2
        time_MOIWOA = t5 - t4
        total_time = time_initial_heuristic + time_MOSA + time_MOIWOA

        # compute metrics
        pareto_solutions = solver.return_pareto_solutions()
        objective_values = solver.return_objectives()

        NOS = len(pareto_solutions)
        MID = compute_MID(problem_params, objectives=objective_values)
        distance = compute_distance(problem_params, objectives=objective_values)

        # save profiling and evaluation metrics
        time_results.append([run_id, time_initial_heuristic, time_MOSA, time_MOIWOA, total_time])
        NOS_results.append([run_id, NOS])
        MID_results.append([run_id, MID])
        distance_results.append([run_id, distance])

        print("Done.")

    # write profiling to csv files
    with open(output_time, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(time_results)

    with open(output_NOS, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(NOS_results)

    with open(output_MID, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(MID_results)

    with open(output_distance, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(distance_results)
