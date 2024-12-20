import sys
import os

library_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if library_path not in sys.path:
    sys.path.append(library_path)

from WCO_lib.dataset import generate_dataset
from WCO_lib.params import ProblemParams, MosaMoiwoaSolverParams
from WCO_lib.solve_moiwoa import MosaMoiwoaSolver
from WCO_lib.evaluate import compute_MID, compute_RASO, compute_distance


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

    # set MOIWOA maximum number of iterations
    MOIWOA_max_iter = 100

    # generate dataset
    generate_dataset(data_dir, bounds_c, bounds_d, bounds_t, bounds_cv, bounds_G)

    print("Dataset generated.\n")

    # load problem parameters
    problem_params = ProblemParams()
    problem_params.load_from_dir(data_dir)
    problem_params.build_graph()

    print("Parameters loaded.\n")

    # set problem
    solver_params = MosaMoiwoaSolverParams()
    solver_params.MOIWOA_max_iter = MOIWOA_max_iter
    solver = MosaMoiwoaSolver(problem_params, solver_params)

    print("Problem set.\n")

    solver.generate_initial_solutions()
    initial_solutions = solver.return_pareto_solutions(stage="initial")

    print("Initial solutions generated:")
    print(initial_solutions)
    print("\n")

    solver.apply_MOSA()
    MOSA_solutions = solver.return_pareto_solutions(stage="MOSA")

    print("MOSA applied, with solutions:")
    print(MOSA_solutions)
    print("\n")

    solver.apply_MOIWOA()
    final_solutions = solver.return_pareto_solutions(stage="final")

    print("Problem solved with MOIWOA, with final solutions:")
    print(final_solutions)
    print("\n")

    # compute metrics
    objectives = solver.return_objectives(stage="final")

    MID = compute_MID(problem_params, objectives=objectives)
    RASO = compute_RASO(problem_params, objectives=objectives)
    distance = compute_distance(problem_params, objectives=objectives)

    print("Evaluation metrics:")
    print("\tNumber of Pareto solutions: ", len(final_solutions))
    print("\tMID: ", MID)
    print("\tRASO: ", RASO)
    print("\tDistance: ", distance)

    print("Objectives:")
    for obj in objectives:
        print(obj)
