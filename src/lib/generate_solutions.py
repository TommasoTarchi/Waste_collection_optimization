from .params import Params, Solution


def generate_initial_solutions(params: Params, num_periods: int) -> Solution:
    """
    Generate initial solution to the problem according to first heuristic
    for a certain number of periods.
    """

    # check if number of periods is valid
    assert num_periods > 0, "Number of periods must be greater than 0."

    solutions_heuristic = []

    # generate solutions for each period with heuristic
    for _ in range(num_periods):
        solution = Solution()
        solution.init_heuristic(params)
        solutions_heuristic.append(solution)
