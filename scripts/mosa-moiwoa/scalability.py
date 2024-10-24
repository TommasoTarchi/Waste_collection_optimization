#
# Run scalability study for MOSA-MOIWOA.
#
# The parameters to be used can be set by the user through a JSON.
# Statistics is done over a number of runs for each problem.
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

from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
from WCO_lib.evaluate import compute_MID, compute_distance


if __name__ == "__main__":

    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver_params_file",
                        type=str,
                        default=None,
                        required=False)
    parser.add_argument("--runs_per_problem",
                        type=int,
                        default=1)

    args = parser.parse_args()

    # get solver parameters
    solver_params = MosaMoiwoaSolverParams()

    if args.solver_params_file is None:
        print("MOSA-MOIWOA parameters file not provided - using default parameters.")
    else:
        print(f"Using MOSA-MOIWOA parameters from file: {args.solver_params_file}")
        solver_params.load_from_file(args.solver_params_file)

    # get number of runs for statistics
    runs_per_problem = args.runs_per_problem

    # set data directories
    data_dir = "../datasets/scalability"
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
    profiling = [["problem_id", "time_initial_heuristic", "time_MOSA", "time_MOIWOA", "total_time"]]
    metrics = [["problem_id", "NOS", "MID", "distance"]]

    # iterate over problems
    for data_path, problem_id in zip(dir_list_paths, dir_list_sorted):

        # load problem parameters and build graph
        problem_params = ProblemParams()
        problem_params.load_from_dir(data_path)
        problem_params.build_graph()

        print(f"Solving problem {problem_id}...")

        # iterate for statistics
        for run_id in range(runs_per_problem):
            print(f"Run {run_id + 1} of {runs_per_problem}...", end=" ", flush=True)

            # set solver for problem
            solver = MosaMoiwoaSolver(problem_params, solver_params)

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

            pareto_solutions = solver.return_pareto_solutions(stage="final")
            objective_values = solver.return_objectives(stage="final")

            # compute metrics
            NOS = len(pareto_solutions)
            MID = compute_MID(problem_params, objectives=objective_values)
            distance = compute_distance(problem_params, objectives=objective_values)

            # compute profiling
            time_initial_heuristic = t1 - t0
            time_MOSA = t3 - t2
            time_MOIWOA = t5 - t4
            total_time = time_initial_heuristic + time_MOSA + time_MOIWOA

            # save pareto solutions
            output_solution_file = os.path.join(output_dir, problem_id + "_solutions_run" + str(run_id) + ".txt")

            with open(output_solution_file, "w") as f:
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

                f.write("\nNUMBER OF ITERATIONS FOR STATISTICS: " + str(runs_per_problem) + "\n")

                f.write("\nPROFILING:\n")
                f.write("Time for initial solutions generation: " + str(time_initial_heuristic) + " seconds\n")
                f.write("Time for MOSA: " + str(time_MOSA) + " seconds\n")
                f.write("Time for MOIWOA: " + str(time_MOIWOA) + " seconds\n")
                f.write("Total time for resolution: " + str(time_initial_heuristic + time_MOSA + time_MOIWOA) + " seconds\n")

                f.write("\nEDGES LIST: " + str(problem_params.existing_edges) + "\n")
                f.write("REQUIRED EDGES LIST: " + str(problem_params.required_edges) + "\n")

                f.write("\nEVALUATION METRICS:\n")
                f.write("Number of Pareto solutions: " + str(len(pareto_solutions)) + "\n")
                f.write("MID for Pareto solutions: " + str(compute_MID(problem_params, objectives=objective_values)) + "\n")
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

            # save profiling and evaluation metrics
            profiling.append([problem_id, time_initial_heuristic, time_MOSA, time_MOIWOA, total_time])

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

    # compute average time
    problem_ids_unique = [problem_ids[i] for i in range(0, len(problem_ids), runs_per_problem)]
    time_avgs = [sum(time_totals[i:i + runs_per_problem]) / runs_per_problem
                 for i in range(0, len(time_totals), runs_per_problem)]

    plt.plot(problem_ids_unique, time_avgs, marker='o', linestyle='-', color='b')
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

    # compute average metrics
    NOS_avgs = [sum(NOS_values[i:i + runs_per_problem]) / runs_per_problem
                for i in range(0, len(NOS_values), runs_per_problem)]
    MID_avgs = [sum(MID_values[i:i + runs_per_problem]) / runs_per_problem
                for i in range(0, len(MID_values), runs_per_problem)]
    distance_avgs = [sum(distance_values[i:i + runs_per_problem]) / runs_per_problem
                     for i in range(0, len(distance_values), runs_per_problem)]

    plt.plot(problem_ids_unique, NOS_avgs, marker='o', linestyle='-', color='b', label='Number of Pareto solutions')
    plt.xlabel('Problem ID')
    plt.ylabel('Number of Pareto solutions')
    plt.title("Average number of Pareto solutions over" + str(runs_per_problem) + " runs")

    plt.savefig(os.path.join(output_dir, "plots", "NOS.png"))
    plt.close()

    plt.plot(problem_ids_unique, MID_avgs, marker='o', linestyle='-', color='b', label='MID')
    plt.xlabel('Problem ID')
    plt.ylabel('MID')
    plt.title("Average MID over" + str(runs_per_problem) + " runs")

    plt.savefig(os.path.join(output_dir, "plots", "MID.png"))
    plt.close()

    plt.plot(problem_ids_unique, distance_avgs, marker='o', linestyle='-', color='b', label='Distance')
    plt.xlabel('Problem ID')
    plt.ylabel('Distance')
    plt.title("Average distance over" + str(runs_per_problem) + " runs")

    plt.savefig(os.path.join(output_dir, "plots", "Distance.png"))
    plt.close()
