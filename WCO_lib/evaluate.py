from math import sqrt

from .params import ProblemParams
from .models_exact import (compute_objective0,
                           compute_objective1,
                           compute_objective2,
                           compute_objective3)


def compute_MID(params: ProblemParams,
                solutions: list) -> float:
    """
    Function to compute mean of ideal distance (MID) of a set of solutions.
    """
    NOS = len(solutions)  # number of solutions
    MID = 0.0

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

        # add to MID
        MID += sqrt(obj0 ** 2 + obj1 ** 2 + obj2 ** 2 + obj3 ** 2)

    return MID / NOS


# TODO: controllare
def compute_RASO(params: ProblemParams,
                 solutions: list) -> float:
    """
    Function to compute rate of achievement to several objectives (RASO) of a set of solutions.
    """
    NOS = len(solutions)  # number of solutions
    RASO = 0.0

    # compute objectives for each solution
    objectives = []
    for solution in solutions:
        objectives.append([compute_objective0(params.theta,
                                              params.c,
                                              params.cv,
                                              params.existing_edges,
                                              solution["x"],
                                              solution["u"]),
                           compute_objective1(params.G,
                                              params.existing_edges,
                                              solution["x"]),
                           compute_objective2(params.sigma,
                                              solution["u"]),
                           compute_objective3(params.T_max,
                                              params.num_vehicles,
                                              params.num_periods,
                                              solution["WT"])])

    # compute terms
    for obj in objectives:
        min_obj = min(obj)
        RASO += sum(obj) / min_obj - 4

    return RASO / NOS


# TODO: controllare
def compute_distance(params: ProblemParams,
                     solutions: list) -> float:
    """
    Function to compute distancing (D) of a set of solutions.
    """
    D = 0.0

    # compute objectives for each solution
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

    print(min(obj0), max(obj0))
    print(min(obj1), max(obj1))
    print(min(obj2), max(obj2))
    print(min(obj3), max(obj3))

    return sqrt(D)
