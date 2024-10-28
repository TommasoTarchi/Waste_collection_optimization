#
# Run scalability study for epsilon-constraint algorithm.
#
# Results saved in ./results/scalability.
#
# The number of epsilon values to be used can be set by the user.
# Also a time limit (in seconds) is set to avoid "endless" computations.
#

import argparse
import sys
import os
import csv
import time
import matplotlib.pyplot as plt

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.params import ProblemParams
from WCO_lib.solve_epsilon import EpsilonSolver
from WCO_lib.evaluate import compute_MID, compute_distance


if __name__ == "__main__":

    # get number of epsilon values
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epsilon", type=int, default=10)
    parser.add_argument("--time_limit", type=float, default=1200.)

    args = parser.parse_args()

    num_epsilon = args.num_epsilon
    time_limit = args.time_limit

    print(f"Using epsilon-solver with number of epsilon values = {num_epsilon}.")
    print(f"Time limit for computation is set to {time_limit} seconds.")

    # set data directories
    data_dir = "../datasets"
    output_dir = "./results/scalability"

    # get all files in the directory
    dir_list = []
    for entry in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, entry)):
            dir_list.append(entry)

    # sort the directories
    dir_list_sorted = sorted(dir_list, key=lambda x: int(x))

    dir_list_paths = []
    for _dir in dir_list_sorted:
        dir_list_paths.append(os.path.join(data_dir, _dir))

    # initialize lists for profiling and evaluation metrics
    profiling = [["problem_id", "time_single_obj", "time_comp_epsilon", "time_multi_obj", "total_time"]]
    metrics = [["problem_id", "NOS", "MID", "distance"]]

    # iterate over problems
    for data_path, problem_id in zip(dir_list_paths, dir_list_sorted):
        # get problem prameters
        params = ProblemParams()
        params.load_from_dir(data_path)

        print(f"Solving problem {problem_id}...", end=" ", flush=True)

        # set solver for problem
        solver = EpsilonSolver(params)

        # solve single-objective problems
        t0 = time.perf_counter()
        time_limit_exceeded = solver.solve_single_objectives(time_limit=time_limit)
        t1 = time.perf_counter()

        # check if time limit was exceeded
        if time_limit_exceeded:
            print(f"Time limit exceeded for problem {problem_id} on single-objective optimization. Exiting.")
            break

        # compute epsilon values
        t2 = time.perf_counter()
        solver.compute_epsilon(num_epsilon)
        t3 = time.perf_counter()

        # solve multi-objective problem
        t4 = time.perf_counter()
        time_limit_exceeded = solver.solve_multi_objective(time_limit=time_limit)
        t5 = time.perf_counter()

        # check if time limit was exceeded
        if time_limit_exceeded:
            print(f"Time limit exceeded for problem {problem_id} on epsilon-constraint optimization. Exiting.")
            break

        pareto_solutions = solver.return_pareto_solutions()

        # compute metrics
        NOS = len(pareto_solutions)
        MID = compute_MID(params, solutions=pareto_solutions)
        distance = compute_distance(params, solutions=pareto_solutions)

        # compute profiling
        time_single_obj = t1 - t0
        time_comp_epsilon = t3 - t2
        time_multi_obj = t5 - t4
        total_time = time_multi_obj + time_single_obj + time_comp_epsilon

        # save profiling and evaluation metrics
        profiling.append([problem_id, time_single_obj, time_comp_epsilon, time_multi_obj, total_time])

        metrics.append([problem_id, NOS, MID, distance])

        print("Done.")

    # write profiling and evaluation metrics to csv files
    with open(os.path.join(output_dir, "profiling.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(profiling)

    with open(os.path.join(output_dir, "metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics)

    # plot total solution time
    rows = profiling[1:]
    problem_ids = [row[0] for row in rows]
    time_totals = [row[4] for row in rows]

    plt.plot(problem_ids, time_totals, marker='o', linestyle='-', color='b')
    plt.xlabel('Problem ID')
    plt.ylabel('Total Time to solution (s)')
    plt.title('Total solution time')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "plots", "solution_time.png"))
    plt.close()

    # plot evaluation metrics
    rows = metrics[1:]

    NOS_values = [row[1] for row in rows]
    MID_values = [row[2] for row in rows]
    distance_values = [row[3] for row in rows]

    plt.plot(problem_ids, NOS_values, marker='o', linestyle='-', color='b', label='Number of Pareto solutions')
    plt.xlabel('Problem ID')
    plt.ylabel('Number of Pareto solutions')
    plt.title('Number of Pareto solutions')

    plt.savefig(os.path.join(output_dir, "plots", "NOS.png"))
    plt.close()

    plt.plot(problem_ids, MID_values, marker='o', linestyle='-', color='b', label='MID')
    plt.xlabel('Problem ID')
    plt.ylabel('MID')
    plt.title('MID')

    plt.savefig(os.path.join(output_dir, "plots", "MID.png"))
    plt.close()

    plt.plot(problem_ids, distance_values, marker='o', linestyle='-', color='b', label='Distance')
    plt.xlabel('Problem ID')
    plt.ylabel('Distance')
    plt.title('Distance')

    plt.savefig(os.path.join(output_dir, "plots", "distance.png"))
    plt.close()
