import gurobipy as gb
import numpy as np

from .params import ProblemParams


class EpsilonSolver:
    """
    Class for Epsilon-constraint solver.
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        self.model = gb.Model()
        self.problem_params = problem_params
        self.x = None
        self.y = None
        self.u = None
        self.LT = None
        self.UT = None
        self.WT = None

    def set_problem(self):
        """
        Set the model of the problem, with objective functions.
        and constraints.
        """
        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # compute maximum number of trips per period (assumed to be equal
        # to the number of required edges in that period)
        P = []
        for req_edges in self.problem_params.required_edges:
            P.append(len(req_edges))

        # rename sets for convenience
        V = self.problem_params.num_nodes
        K = self.problem_params.num_vehicles
        T = self.problem_params.num_periods

        # set variables (same indexes and names as in the paper)
        self.x = self.model.addVars([(i, j, k, p, t) for i in range(V)
                                     for j in range(V)
                                     for k in range(K)
                                     for t in range(T)
                                     for p in range(P[t])],
                                    vtype=gb.GRB.INTEGER,
                                    lb=0,
                                    name="x")

        self.y = self.model.addVars([(i, j, k, p, t) for t in range(T)
                                     for (i, j) in self.problem_params.required_edges[t]
                                     for k in range(K)
                                     for p in range(P[t])],
                                    vtype=gb.GRB.BINARY,
                                    name="y")

        self.u = self.model.addVars([(k, t) for k in range(K)
                                     for t in range(T)],
                                    vtype=gb.GRB.BINARY,
                                    name="u")

        self.LT = self.model.addVars([(k, p, t) for t in range(T)
                                      for p in range(P[t])
                                      for k in range(K)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="LT")

        self.UT = self.model.addVars([(k, p, t) for t in range(T)
                                      for p in range(P[t])
                                      for k in range(K)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="UT")

        self.WT = self.model.addVars([(k, t) for k in range(K)
                                      for t in range(T)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="WT")

        # set constraints for relations between variables
        # compute y
        # compute u
        # compute WT

        # set proper constraints

        # set objectives
