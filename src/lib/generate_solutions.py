import numpy as np

from .params import Params, SinglePeriodSolution


def generate_heuristic_solution(params: Params):
    """
    Generate an initial solutions to the problem according to the first heuristic
    in the paper.
    """

    solution_heuristic = []

    # generate solutions for each period with heuristic
    for period in range(params.num_periods):
        solution = SinglePeriodSolution(period)
        solution.init_heuristic(params)
        solution_heuristic.append(solution)

    return solution_heuristic


def MOSA(T_0: int, params: Params, initial_solution: list):
    """
    Apply Multi-Objective Simulated Annealing (MOSA) to an initial solution.
    """

    current_solution = initial_solution

    # compute objective functions for the current solution
    current_objective_functions = [0, 0, 0, 0]
    for period_solution in current_solution:
        current_objective_functions += period_solution.compute_objective_functions(params)

    while ...:  # ADD STOPPING CRITERIA

        ngbr_solution = []

        # generate neighbor solution
        for period in range(params.num_periods):

            # get first and second part of the initial period solution
            current_first_part = current_solution[period].first_part.copy()
            current_second_part = current_solution[period].second_part.copy()

            ngbr_solution.append(SinglePeriodSolution(period))

            # generate neighbor period solution
            if np.random.uniform() < 0.5:
                # perturb first part
                ngbr_solution[period].set_first_part(np.random.permutation(current_first_part.size[0]))

                # copy second part
                ngbr_solution[period].set_second_part(current_second_part)

                # adjust second part to satisfy constraints
                ngbr_solution[period].adjust_second_part(params)

            else:
                # copy first part
                ngbr_solution[period].set_first_part(current_first_part)

                # perturb second part
                vehicle_to_substitute = np.random.choice(current_second_part)
                new_vehicle = np.random.randint(params.num_vehicles)
                ngbr_second_part = current_second_part.copy()
                ngbr_second_part[ngbr_second_part == vehicle_to_substitute] = new_vehicle
                ngbr_solution[period].set_second_part(ngbr_second_part)

                # adjust first part to satisfy constraints
                ngbr_solution[period].adjust_first_part(params)

        # compute objective functions for the neighbor solution
        # TODO

        # compute acceptance probability
        # TODO

        # accept or reject neighbor solution
        # TODO (remember to recompute current objective functions if neighbor is accepted)

        # time annealing
        # TODO
