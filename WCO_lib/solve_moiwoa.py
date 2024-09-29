import numpy as np
from deap import creator, base, tools

from .params import ProblemParams, MosaMoiwoaSolverParams
from .models_heuristic import generate_heuristic_solution, MOSA


# create DEAP classes for non-dominated sorting
creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMinMulti)


def compute_fitness(objectives: np.ndarray, avg_objectives: np.ndarray) -> np.float64:
    """
    Compute the fitness value of a solution.
    """
    fitness = 0.0
    fitness = np.sum(objectives / avg_objectives) / objectives.shape[0]

    return fitness


def compute_n_seeds(fitness: float,
                    min_fitness: float,
                    max_fitness: float,
                    S_min: float,
                    S_max: float) -> int:
    """
    Compute the number of seeds for a solution.
    """
    S = int(S_min + (S_max - S_min) * (fitness - min_fitness) / (min_fitness - max_fitness))

    return round(S)


def sort_seeds(objective_values: list) -> list:
    """
    Sort the seed population with non-dominated sorting and crowding distance.
    """
    sorted_seeds_idx = []

    # Create DEAP individuals and assign fitness values
    individuals = [creator.Individual(objectives) for objectives in objective_values]
    for ind in individuals:
        ind.fitness.values = tuple(ind)

    # Apply non-dominated sorting
    fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=False)

    # sort seeds by fronts and crowding distance
    for front in fronts:
        if len(front) == 1:
            front_idx = [np.where(np.all(np.array(individuals) == fit, axis=1))[0][0] for fit in front]
            sorted_seeds_idx.append(front_idx[0])

        elif len(front) > 1:
            # compute crowding distance
            tools.emo.assignCrowdingDist(front)
            for ind in front:
                if np.isinf(ind.fitness.crowding_dist):
                    ind.fitness.crowding_dist = 1e6

            # sort front by crowding distance
            front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)

            # Extract sorted indices
            front_sorted = [np.where(np.all(np.array(individuals) == fit, axis=1))[0][0] for fit in front]
            sorted_seeds_idx.extend(front_sorted)

    return sorted_seeds_idx


def MOIWOA(initial_seeds: list,
           problem_params: ProblemParams,
           S_min: float = 9.0,
           S_max: float = 200.0,
           N_max: int = 100,
           max_iter: int = 300) -> list:
    """
    Apply Multi-Objective Invasive Weed Optimization Algorithm (MOIWOA) to a list of
    initial solutions.
    """
    current_seeds = initial_seeds
    current_objectives_values = []
    current_fitness_values = []

    # compute objectives values for initial seeds
    for seed in current_seeds:
        seed_objectives = np.zeros(4)
        for period_solution in seed:
            period_solution.compute_objectives(problem_params)
            seed_objectives += period_solution.objectives
        current_objectives_values.append(seed_objectives)
    avg_objectives = np.zeros(4)
    for objectives in current_objectives_values:
        avg_objectives += objectives
    avg_objectives /= len(current_objectives_values)  # for standardization

    # compute fitness values for initial seeds
    for seed_objectives in current_objectives_values:
        current_fitness_values.append(compute_fitness(seed_objectives, avg_objectives))
    min_fitness = min(current_fitness_values)
    max_fitness = max(current_fitness_values)

    n_iter = 0
    while n_iter < max_iter:

        # compute average objective values on current seeds (for standardization)
        avg_objectives = np.zeros(4)
        for objectives in current_objectives_values:
            avg_objectives += objectives
        avg_objectives /= len(current_objectives_values)

        # generate children seeds
        new_seeds = []
        for seed, fitness in zip(current_seeds, current_fitness_values):
            n_children = compute_n_seeds(fitness, min_fitness, max_fitness, S_min, S_max)

            # TODO: produce children seeds and add them to new_seeds

        # compute objectives and fitness values for children seeds
        new_fitness_values = []
        new_objectives_values = []
        for seed in new_seeds:
            seed_objectives = np.zeros(4)
            for period_solution in seed:
                period_solution.compute_objectives(problem_params)
                seed_objectives += period_solution.objectives
            new_objectives_values.append(seed_objectives)
            new_fitness_values.append(compute_fitness(seed_objectives, avg_objectives))

        # add children seeds to solutions
        current_seeds += new_seeds
        current_objectives_values += new_objectives_values
        current_fitness_values += new_fitness_values

        # truncate seed population if larger than upper limit
        if len(current_seeds) > N_max:
            # apply non-dominated sorting
            sorted_seeds_idx = sort_seeds(current_fitness_values)

            # truncate seed population
            current_seeds = [current_seeds[i] for i in sorted_seeds_idx[:N_max]]
            current_objectives_values = [current_objectives_values[i] for i in sorted_seeds_idx[:N_max]]
            current_fitness_values = [current_fitness_values[i] for i in sorted_seeds_idx[:N_max]]

        # compute min and max fitness values
        min_fitness = min(current_fitness_values)
        max_fitness = max(current_fitness_values)

        n_iter += 1

    return current_seeds


class MosaMoiwoaSolver:
    """
    Solver class to solve the optimization problem using MOSA-MOIWOA.
    """

    def __init__(self, problem_params: ProblemParams, solver_params: MosaMoiwoaSolverParams) -> None:
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
                                      self.problem_params,
                                      self.solver_params.MOIWOA_S_min,
                                      self.solver_params.MOIWOA_S_max,
                                      self.solver_params.MOIWOA_N_max,
                                      self.solver_params.MOIWOA_max_iter)
