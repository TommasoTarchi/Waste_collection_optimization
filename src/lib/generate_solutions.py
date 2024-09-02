import numpy as np

from .params import ProblemParams, SinglePeriodSolution


def generate_heuristic_solution(params: ProblemParams) -> list:
    """
    Generate a (single) initial solution to the problem according to the first heuristic
    in the paper.
    """
    solution_heuristic = []

    # generate solutions for each period with heuristic
    for period in range(params.num_periods):
        solution = SinglePeriodSolution(period)
        solution.init_heuristic(params)
        solution_heuristic.append(solution)

    return solution_heuristic


def dominates(objective_functions_1: list, objective_functions_2: list) -> bool:
    """
    Check if objective_functions_1 dominates objective_functions_2.
    """
    assert len(objective_functions_1) == len(objective_functions_2), "Objective functions must have the same length."

    for i in range(len(objective_functions_1)):
        if objective_functions_1[i] < objective_functions_2[i]:
            return False

    for i in range(len(objective_functions_1)):
        if objective_functions_1[i] > objective_functions_2[i]:
            return True

    return False


def geometric_cooling(T_old: float, alpha: float) -> float:
    """
    Apply geometric cooling to the temperature.
    """
    assert 0 < alpha < 1, "Alpha must be between 0 and 1."

    T_new = alpha * T_old

    return T_new


def acceptance_probability(current_objective_functions: list, ngbr_objective_functions: list, T: float, K: float) -> float:
    """
    Compute the acceptance probability for a non-dominant neighbor solution.
    """
    assert len(current_objective_functions) == len(ngbr_objective_functions), "Objective functions must have the same length."

    # compute average difference between objective functions
    diff = 0
    for i in range(len(current_objective_functions)):
        diff += current_objective_functions[i] - ngbr_objective_functions[i]
    diff /= len(current_objective_functions)

    # compute acceptance probability
    prob = np.exp(-diff / (K * T))

    return min(1, prob)


def MOSA(initial_solution: list,
         params: ProblemParams,
         T_0: float = 800.0,
         max_iter: int = 200,
         max_non_improving_iter: int = 10,
         alpha: float = 0.9,
         K: float = 70.0) -> list:
    """
    Apply Multi-Objective Simulated Annealing (MOSA) to a (single) initial solution.
    """
    T = T_0
    current_solution = initial_solution

    # compute objective functions for the current solution
    current_objective_functions = np.array([0, 0, 0, 0])
    for period_solution in current_solution:
        period_solution.compute_objective_functions(params)
        current_objective_functions += period_solution.objectives

    n_iter = 0  # number of iterations
    non_improving_iter = 0  # number of non-improving iterations
    while n_iter < max_iter and non_improving_iter < max_non_improving_iter:

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
        ngbr_objective_functions = np.array([0, 0, 0, 0])
        for period_solution in ngbr_solution:
            period_solution.compute_objective_functions(params)
            ngbr_objective_functions += period_solution.objectives

        # increase number of non-improving iterations
        non_improving_iter += 1

        # accept neighbor solution if dominant
        if dominates(ngbr_objective_functions, current_objective_functions):
            current_solution = ngbr_solution
            current_objective_functions = ngbr_objective_functions
            non_improving_iter = 0

        # accept neighbor solution with probability if non-dominant
        elif dominates(current_objective_functions, ngbr_objective_functions):
            accept_prob = acceptance_probability(current_objective_functions, ngbr_objective_functions, T, K)
            if accept_prob > np.random.uniform():
                current_solution = ngbr_solution
                current_objective_functions = ngbr_objective_functions
                non_improving_iter = 0

        # temperature cooling
        T = geometric_cooling(T, alpha=alpha)

        n_iter += 1

    return current_solution
