import numpy as np
from math import sqrt
from deap import creator, base, tools
from typing import Optional, List, Dict

from .params import ProblemParams


# TODO: controllare tutte le funzioni per evaluation metrics


# create DEAP classes for non-dominated sorting
creator.create("FitnessMinMulti", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMinMulti)


def sort_solutions(objective_values: list) -> tuple:
    """
    Sort the solutions (i.e. seeds) population with non-dominated sorting and
    crowding distance.
    Returns a tuple containing:
    1. List of sorted indices for all seeds
    2. List of indices for seeds in the first Pareto front
    """
    sorted_solutions_idx = []
    first_front_idx = []

    # create DEAP individuals and assign fitness values
    individuals = [creator.Individual(objectives) for objectives in objective_values]
    for ind in individuals:
        ind.fitness.values = tuple(ind)

    # apply non-dominated sorting
    fronts = tools.sortNondominated(individuals, len(individuals), first_front_only=False)

    # sort solutions by fronts and crowding distance
    for i, front in enumerate(fronts):
        if len(front) == 1:
            front_idx = [np.where(np.all(np.array(individuals) == fit, axis=1))[0][0] for fit in front]
            sorted_solutions_idx.append(front_idx[0])

        elif len(front) > 1:
            # compute crowding distance
            tools.emo.assignCrowdingDist(front)
            for ind in front:
                if np.isinf(ind.fitness.crowding_dist):
                    ind.fitness.crowding_dist = 1e6

            # sort front by crowding distance
            front.sort(key=lambda ind: ind.fitness.crowding_dist, reverse=True)

            # extract sorted indices
            front_sorted = [np.where(np.all(np.array(individuals) == fit, axis=1))[0][0] for fit in front]
            sorted_solutions_idx.extend(front_sorted)

            # add indices of seeds in the first front
            if i == 0:
                first_front_idx.extend(front_sorted)

    return sorted_solutions_idx, first_front_idx


def compute_objective0(theta: float,
                       c: np.ndarray,
                       cv: np.ndarray,
                       existing_edges: list,
                       x: np.ndarray,
                       u: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_0 (Z_1 in the original paper).

    Assume input data in array format from epsilon-constraint method.
    """
    obj = 0

    # first term
    partial_sums = np.sum(x, axis=(0, 1, 2))
    for (count, (i, j)) in enumerate(existing_edges):
        obj += c[i, j] * partial_sums[count]
    obj *= theta

    # second term
    obj += cv @ np.sum(u, axis=1)

    return obj


def compute_objective1(G: np.ndarray, existing_edges: list, x: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_1 (Z_2 in the original paper).

    Assume input data in array format from epsilon-constraint method.
    """
    obj = 0
    partial_sums = np.sum(x, axis=(0, 1, 2))
    for (count, (i, j)) in enumerate(existing_edges):
        obj += G[i, j] * partial_sums[count]

    return obj


def compute_objective2(sigma: float, u: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_2 (Z_3 in the original paper).

    Assume input data in array format from epsilon-constraint method.
    """
    return sigma * np.sum(u)


def compute_objective3(T_max: float,
                       num_vehicles: int,
                       num_periods: int,
                       WT: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_3 (Z_4 in the original paper).

    Assume input data in array format from epsilon-constraint method.
    """
    return num_vehicles * num_periods - float(np.sum(WT)) / T_max


def compute_normalized_MID(params: ProblemParams,
                           solutions: Optional[Dict] = None,
                           objectives: Optional[List] = None) -> float:
    """
    Function to compute mean of ideal distance (MID) of a set of solutions.

    The metric can be computed starting from either the solution or the pre-computed
    objectives.
    """
    if ((objectives is not None and solutions is not None)
        or (objectives is None and solutions is None)):
        raise ValueError("Either solutions or objectives must be provided.")

    NOS = 0  # number of solutions
    if objectives is not None:
        NOS = len(objectives)
    elif solutions is not None:
        NOS = len(solutions)

    MID = 0.0

    # if objects provided
    if objectives is not None:
        for obj in objectives:
            # compute maximum value for rescaling
            max_obj = np.max(obj)

            # add minimization objectives to MID
            MID += (obj[0] / max_obj) ** 2 + (obj[1] / max_obj) ** 2 + (obj[3] / max_obj) ** 2

            # add maximization objective to MID
            MID += ((params.sigma * params.num_vehicles * params.num_periods - obj[2]) / max_obj) ** 2

    # if solutions provided
    elif solutions is not None:
        for solution in solutions:
            # get solution values
            x = solution["x"]
            u = solution["u"]
            WT = solution["WT"]

            # compute objectives
            obj0 = compute_objective0(params.theta,
                                      params.c,
                                      params.cv,
                                      params.existing_edges,
                                      x,
                                      u)
            obj1 = compute_objective1(params.G, params.existing_edges, x)
            obj2 = compute_objective2(params.sigma, u)
            obj3 = compute_objective3(params.T_max,
                                      params.num_vehicles,
                                      params.num_periods,
                                      WT)

            # compute maximum value for rescaling
            max_obj = max(obj0,
                          obj1,
                          params.sigma * params.num_vehicles * params.num_periods - obj2,
                          obj3)

            # add minimization objectives to MID
            MID += (obj0 / max_obj) ** 2 + (obj1 / max_obj) ** 2 + (obj3 / max_obj) ** 2

            # add maximization objective to MID
            MID += ((params.sigma * params.num_vehicles * params.num_periods - obj2) / max_obj) ** 2

    return sqrt(MID) / NOS


def compute_RASO(params: ProblemParams,
                 solutions: Optional[Dict] = None,
                 objectives: Optional[List] = None) -> float:
    """
    Function to compute rate of achievement to several objectives (RASO) of a set of solutions.

    The metric can be computed starting from either the solution or the pre-computed
    objectives.
    """
    if ((objectives is not None and solutions is not None)
        or (objectives is None and solutions is None)):
        raise ValueError("Either solutions or objectives must be provided.")

    NOS = 0  # number of solutions
    if objectives is not None:
        NOS = len(objectives)
    elif solutions is not None:
        NOS = len(solutions)

    RASO = 0.0
    objectives_list = []

    # if objects provided, get them
    if objectives is not None:
        objectives_list = objectives

        for obj in objectives_list:
            obj[2] += 1

    # if solutions provided, compute objectives
    elif solutions is not None:
        for solution in solutions:
            objectives_list.append([compute_objective0(params.theta,
                                                       params.c,
                                                       params.cv,
                                                       params.existing_edges,
                                                       solution["x"],
                                                       solution["u"]),
                                    compute_objective1(params.G,
                                                       params.existing_edges,
                                                       solution["x"]),
                                    1 + params.sigma * params.num_vehicles * params.num_periods - compute_objective2(
                                        params.sigma,
                                        solution["u"],
                                        ),
                                    compute_objective3(params.T_max,
                                                       params.num_vehicles,
                                                       params.num_periods,
                                                       solution["WT"])])

    # compute terms
    for obj in objectives_list:
        min_obj = min(obj)

        # correct for division by zero
        for i in range(len(obj)):
            if obj[i] == 0:
                obj[i] = 1e-6

        RASO += sum(obj) / min_obj - 4

    return RASO / NOS


def compute_distance(params: ProblemParams,
                     solutions: Optional[Dict] = None,
                     objectives: Optional[List] = None) -> float:
    """
    Function to compute distancing (D) of a set of solutions.

    The metric can be computed starting from either the solution or the pre-computed
    objectives.
    """
    if ((objectives is not None and solutions is not None)
        or (objectives is None and solutions is None)):
        raise ValueError("Either solutions or objectives must be provided.")

    D = 0.0
    obj0 = []
    obj1 = []
    obj2 = []
    obj3 = []

    # if objects provided, get them
    if objectives is not None:
        for obj in objectives:
            obj0.append(obj[0])
            obj1.append(obj[1])
            obj2.append(obj[2])
            obj3.append(obj[3])

    # if solutions provided, compute objectives
    elif solutions is not None:
        obj0 = [compute_objective0(params.theta,
                                   params.c,
                                   params.cv,
                                   params.existing_edges,
                                   solution["x"],
                                   solution["u"]) for solution in solutions]

        obj1 = [compute_objective1(params.G,
                                   params.existing_edges,
                                   solution["x"]) for solution in solutions]

        obj2 = [compute_objective2(params.sigma,
                                   solution["u"]) for solution in solutions]

        obj3 = [compute_objective3(params.T_max,
                                   params.num_vehicles,
                                   params.num_periods,
                                   solution["WT"]) for solution in solutions]

    # compute distances
    D += (max(obj0) - min(obj0)) ** 2
    D += (max(obj1) - min(obj1)) ** 2
    D += (max(obj2) - min(obj2)) ** 2
    D += (max(obj3) - min(obj3)) ** 2

    return sqrt(D)
