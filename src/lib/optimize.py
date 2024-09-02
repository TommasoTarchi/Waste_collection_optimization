from .params import ProblemParams


def compute_fitness(solution: list) -> float:
    """
    Compute the fitness value of a solution.
    """
    # TODO
    pass


def compute_n_seeds(fitness: float, min_fitness: float, max_fitness: float, S_min: float, S_max: float) -> int:
    """
    Compute the number of seeds for a solution.
    """
    # TODO
    pass


def MOIWOA(initial_seeds: list,
           S_min: float = 9.0,
           S_max: float = 200.0,
           N_max: int = 100,
           max_iter: int = 300) -> list:
    """
    Apply Multi-Objective Invasive Weed Optimization Algorithm.
    """
    current_solutions = []
    N = len(initial_seeds)

    n_iter = 0
    while n_iter < max_iter:

        # compute fitness values
        fitness_values = []
        for seed in initial_seeds:
            fitness_values.append(compute_fitness(seed))
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        # compute number of seeds for each current seed
        n_seeds = []
        for fitness in fitness_values:
            n_seeds.append(compute_n_seeds(fitness, min_fitness, max_fitness, S_min, S_max))

        # TODO

    return current_solutions


class Solver:
    """
    Solver class to solve the optimization problem.
    """

    def __init__(self, params: ProblemParams, n_initial_solutions: int = 10):
        self.params = params
        self.n_initial_solutions = n_initial_solutions

    # init solutions with heuristic

    # apply MOSA to initial solutions

    # apply MOIWOA to obtained solutions
