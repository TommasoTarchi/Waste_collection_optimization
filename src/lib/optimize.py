from deap.tools import sortNondominated

from .params import ProblemParams, SolverParams
from .generate_solutions import generate_heuristic_solution, MOSA


def compute_fitness() -> float:
    """
    Compute the fitness value of a solution.
    """
    # TODO
    pass


def compute_n_seeds(fitness: float, min_fitness: float, max_fitness: float, S_min: float, S_max: float) -> int:
    """
    Compute the number of seeds for a solution.
    """
    S = int(S_min + (S_max - S_min) * (fitness - min_fitness) / (min_fitness - max_fitness))

    return round(S)


def sort_nondominated(objective_values: list) -> list:
    """
    Sort the seed population with non-dominated sorting.
    """
    sorted_seeds = []

    # TODO (use deap.tools.sortNondominated)

    return sorted_seeds


def MOIWOA(initial_seeds: list,
           S_min: float = 9.0,
           S_max: float = 200.0,
           N_max: int = 100,
           max_iter: int = 300) -> list:
    """
    Apply Multi-Objective Invasive Weed Optimization Algorithm (MOIWOA) to a list of
    initial solutions.
    """
    current_seeds = initial_seeds

    # compute fitness values
    current_fitness_values = []
    for seed in current_seeds:
        current_fitness_values.append(compute_fitness())
    min_fitness = min(current_fitness_values)
    max_fitness = max(current_fitness_values)

    n_iter = 0
    while n_iter < max_iter:

        # compute children seeds
        new_seeds = []
        new_fitness_values = []
        for seed, fitness in zip(current_seeds, current_fitness_values):
            n_children = compute_n_seeds(fitness, min_fitness, max_fitness, S_min, S_max)

            # TODO: distribute children seeds and add them to new_seeds,
            #       also compute their fitness values and add them to new_fitness_values

        # add children seeds to solutions
        current_seeds += new_seeds
        current_fitness_values += new_fitness_values

        # truncate seed population if larger than upper limit
        if len(current_seeds) > N_max:
            sorted_seeds_idx = sort_nondominated(current_fitness_values)

            # TODO: remove seeds accordingly
            pass

        n_iter += 1

    return current_seeds


class MosaMoiwoaSolver:
    """
    Solver class to solve the optimization problem (MOSA-MOIWOA).
    """

    def __init__(self, problem_params: ProblemParams, solver_params: SolverParams):
        self.problem_params = problem_params
        self.solver_params = solver_params
        self.initial_solutions = None
        self.MOSA_solutions = None
        self.final_solutions = None

    def generate_initial_solutions(self) -> None:
        """
        Generate initial solutions with heuristic.
        """
        self.initial_solutions = []
        for _ in range(self.solver_params.N_0):
            initial_solution = generate_heuristic_solution(self.problem_params)
            self.initial_solutions.append(initial_solution)

    def apply_MOSA(self) -> None:
        """
        Apply Multi-Objective Simulated Annealing (MOSA) to initial solutions.
        """
        assert self.initial_solutions is not None, "Before applying MOSA initial solutions must be generated."

        self.MOSA_solutions = []
        for initial_solution in self.initial_solutions:
            MOSA_solution = MOSA(initial_solution,
                                 self.problem_params,
                                 self.solver_params.MOSA_T_0,
                                 self.solver_params.MOSA_max_iter,
                                 self.solver_params.MOSA_max_non_improving_iter,
                                 self.solver_params.MOSA_alpha,
                                 self.solver_params.MOSA_K)
            self.MOSA_solutions.append(MOSA_solution)

    def apply_MOIWOA(self) -> None:
        """
        Apply Multi-Objective Invasive Weed Optimization Algorithm (MOIWOA) to MOSA solutions.
        """
        assert self.MOSA_solutions is not None, "Before applying MOIWOA MOSA must be applied."

        self.final_solutions = MOIWOA(self.MOSA_solutions,
                                      self.solver_params.MOIWOA_S_min,
                                      self.solver_params.MOIWOA_S_max,
                                      self.solver_params.MOIWOA_N_max,
                                      self.solver_params.MOIWOA_max_iter)
