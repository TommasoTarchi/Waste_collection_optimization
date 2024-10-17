import argparse
import numpy as np
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
from WCO_lib.evaluate import compute_normalized_MID, compute_RASO, compute_distance


if __name__ == "__main__":

    # get scalabilty type
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",
                        type=str,
                        choices=["graph", "epoch", "vehicles"],
                        default=None)
    parser.add_argument("--num_epsilon", type=int, default=4)

    args = parser.parse_args()

    num_epsilon = args.num_epsilon

    # get data directories
    data_dir = None
    output_dir = None
    if args.type == "graph":
        data_dir = os.path.join("./data", "graph_scalability")
        output_dir = "./results/graph_scalability/"
    elif args.type == "epoch":
        data_dir = os.path.join("./data", "epoch_scalability")
        output_dir = "./results/epoch_scalability/"
    elif args.type == "vehicles":
        data_dir = os.path.join("./data", "vehicles_scalability")
        output_dir = "./results/vehicles_scalability/"

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
    profiling = [["size", "time_single_obj", "time_comp_epsilon", "time_multi_obj", "total_time"]]
    metrics = [["size", "normalized_MID", "RASO", "distance"]]

    # iterate over problems
    for data_path, size in zip(dir_list_paths, dir_list_sorted):
        # get problem prameters
        params = ProblemParams()
        params.load_from_dir(data_path)

        # set solver for problem
        solver = EpsilonSolver(params)

        t0 = time.perf_counter()

        # solve single-objective problems
        solver.solve_single_objectives()

        t1 = time.perf_counter()

        # compute epsilon values
        solver.compute_epsilon(num_epsilon)

        t2 = time.perf_counter()

        # solve multi-objective problem
        solver.solve_multi_objective()

        t3 = time.perf_counter()

        pareto_solutions = solver.return_pareto_solutions()

        # compute metrics
        normalized_MID = compute_normalized_MID(params, solutions=pareto_solutions)
        RASO = compute_RASO(params, solutions=pareto_solutions)
        distance = compute_distance(params, solutions=pareto_solutions)

        # compute profiling
        time_single_obj = t1 - t0
        time_comp_epsilon = t2 - t1
        time_multi_obj = t3 - t2
        total_time = t3 - t0

        # save pareto solutions
        output_solution_file = os.path.join(output_dir, size + "_solutions.txt")

        with open(output_solution_file, "w") as f:
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
            f.write("Total time for rsolution: " + str(total_time) + " seconds\n")

            f.write("\nEVALUATION METRICS:\n")
            f.write("Normalized mid for pareto solutions: " + str(normalized_MID) + "\n")
            f.write("Raso for pareto solutions: " + str(RASO) + "\n")
            f.write("Distance for pareto solutions: " + str(distance) + "\n\n")

            f.write("\nEDGES LIST: " + str(params.existing_edges) + "\n")
            f.write("REQUIRED EDGES LIST: " + str(params.required_edges) + "\n")

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

        # save profiling and evaluation metrics
        profiling.append([size, time_single_obj, time_comp_epsilon, time_multi_obj, total_time])

        metrics.append([size, normalized_MID, RASO, distance])

    # write profiling and evaluation metrics to csv files
    with open(os.path.join(output_dir, "profiling.txt"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(profiling)

    with open(os.path.join(output_dir, "metrics.txt"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics)

    # plot total solution time
    rows = profiling[1:]
    sizes = [row[0] for row in rows]
    time_totals = [row[4] for row in rows]

    plt.plot(sizes, time_totals, marker='o', linestyle='-', color='b')
    plt.xlabel('Size')
    plt.ylabel('Total Time to solution (s)')
    plt.title('Total solution time')
    plt.grid(True)

    plt.savefig(os.path.join(output_dir, "plots", "solution_time.png"))
    plt.close()

    # plot evaluation metrics
    rows = metrics[1:]

    normalized_MID_values = [row[1] for row in rows]
    RASO_values = [row[2] for row in rows]
    distance_values = [row[3] for row in rows]

    plt.plot(sizes, normalized_MID_values, marker='o', linestyle='-', color='b', label='Normalized MID')
    plt.xlabel('Size')
    plt.ylabel('Normalized MID')
    plt.title('Normalized MID')

    plt.savefig(os.path.join(output_dir, "plots", "normalized_MID.png"))
    plt.close()

    plt.plot(sizes, RASO_values, marker='o', linestyle='-', color='b', label='RASO')
    plt.xlabel('Size')
    plt.ylabel('RASO')
    plt.title('RASO')

    plt.savefig(os.path.join(output_dir, "plots", "RASO.png"))
    plt.close()

    plt.plot(sizes, distance_values, marker='o', linestyle='-', color='b', label='Distance')
    plt.xlabel('Size')
    plt.ylabel('Distance')
    plt.title('Distance')

    plt.savefig(os.path.join(output_dir, "plots", "distance.png"))
    plt.close()
