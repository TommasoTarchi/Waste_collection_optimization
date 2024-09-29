import numpy as np

from .params import ProblemParams, SinglePeriodVectorSolution


def generate_heuristic_solution(problem_params: ProblemParams) -> list:
    """
    Generate a (single) initial solution to the problem according to the first heuristic
    in the paper.
    """
    solution_heuristic = []

    # generate solutions for each period with heuristic
    for period in range(problem_params.num_periods):
        solution = SinglePeriodVectorSolution(period)
        solution.init_heuristic(problem_params)
        solution_heuristic.append(solution)

    return solution_heuristic


def dominates(target_objective: np.ndarray, comparison_objective: np.ndarray) -> bool:
    """
    Check if the target solution dominates the comparison solution.
    """
    assert target_objective.shape == comparison_objective.shape, "Objective functions must have the same shape."

    if np.all(target_objective >= comparison_objective):
        if np.any(target_objective > comparison_objective):
            return True

    return False


def geometric_cooling(T: float, alpha: float) -> float:
    """
    Apply geometric cooling to the temperature.
    """
    assert T > 0, "Temperature must be positive."
    assert 0 < alpha < 1, "Alpha must be between 0 and 1."

    T_new = alpha * T

    return T_new


def acceptance_probability(current_objective_functions: np.ndarray,
                           ngbr_objective_functions: np.ndarray,
                           T: float,
                           K: float) -> float:
    """
    Compute the acceptance probability for a non-dominant neighbor solution.
    """
    assert T > 0, "Temperature must be positive."
    assert K > 0, "K must be positive."
    assert current_objective_functions.shape == ngbr_objective_functions.shape, "Objective functions must have the same shape."

    # compute average difference between objective functions
    diff = np.mean(current_objective_functions - ngbr_objective_functions)

    # compute acceptance probability
    prob = np.exp(-diff / (K * T))

    return min(1, prob)


def MOSA(initial_solution: list,
         problem_params: ProblemParams,
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
        period_solution.compute_objectives(problem_params)
        current_objective_functions += period_solution.objectives

    n_iter = 0  # number of iterations
    non_improving_iter = 0  # number of non-improving iterations
    while n_iter < max_iter and non_improving_iter < max_non_improving_iter:
        ngbr_solution = []

        # generate neighbor solution
        for period in range(problem_params.num_periods):

            # get first and second part of the initial period solution
            current_first_part = current_solution[period].first_part.copy()
            current_second_part = current_solution[period].second_part.copy()

            ngbr_solution.append(SinglePeriodVectorSolution(period))

            # generate neighbor period solution
            if np.random.uniform() < 0.5:
                # perturb first part
                ngbr_solution[period].set_first_part(np.random.permutation(current_first_part.size[0]))

                # copy second part
                ngbr_solution[period].set_second_part(current_second_part)

                # adjust second part to satisfy constraints
                ngbr_solution[period].adjust_second_part(problem_params)

            else:
                # copy first part
                ngbr_solution[period].set_first_part(current_first_part)

                # perturb second part
                vehicle_to_substitute = np.random.choice(current_second_part)
                new_vehicle = np.random.randint(problem_params.num_vehicles)
                ngbr_second_part = current_second_part.copy()
                ngbr_second_part[ngbr_second_part == vehicle_to_substitute] = new_vehicle
                ngbr_solution[period].set_second_part(ngbr_second_part)

                # adjust first part to satisfy constraints
                ngbr_solution[period].adjust_first_part(problem_params)

        # compute objective functions for the neighbor solution
        ngbr_objective_functions = np.array([0, 0, 0, 0])
        for period_solution in ngbr_solution:
            period_solution.compute_objectives(problem_params)
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
