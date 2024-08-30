from .params import Params, Solution


def generate_initial_solutions(params: Params) -> Solution:
    """
    Generate initial solution to the problem according to first heuristic
    for a single period.
    """

    # Generate random first part
    solution = Solution()

    # compute second part with heuristic
    solution.init_heuristic(params)

    return solution
