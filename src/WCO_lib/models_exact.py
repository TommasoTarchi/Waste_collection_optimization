import gurobipy as gb
import numpy as np

from .params import ProblemParams


class BaseModel:
    """
    Base class for setting variables and constraints of the model.
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        self.problem_params = problem_params
        self.model = None
        self.x = None
        self.y = None
        self.u = None
        self.LT = None
        self.UT = None

        # compute maximum number of trips per period (assumed to be equal
        # to the number of required edges in that period)
        self.P = []
        for req_edges in self.problem_params.required_edges:
            self.P.append(len(req_edges))

        # rename sets for convenience
        self.V = self.problem_params.num_nodes
        self.K = self.problem_params.num_vehicles
        self.T = self.problem_params.num_periods

    def set_up_model(self):
        """
        Set the model of the problem, with objective functions and constraints.
        """
        self.model = gb.Model("Epsilon-constraint")

        # rename sets vars for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

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

        # set model constraints (reference numbers are the same as in the
        # original paper) (constraints (20) and (21) are already included
        # inside the definition of the variables)

        # constraint (5)

        # constraint (6)
        for t in range(T):
            for (i, j) in self.problem_params.required_edges[t]:
                self.model.addConstr(gb.quicksum(self.y[i, j, k, p, t]
                                                 + self.y[j, i, k, p, t] for k in range(K)
                                                 for p in range(P[t])) == 1)

        # precompute linear expression for scalar product between d and y
        dy_linear = []
        for t in range(T):
            dy_linear.append([])
            for p in range(P[t]):
                dy_linear[t].append([])
                for k in range(K):
                    dy_linear[t][p].append(gb.LinExpr([(self.problem_params.d[i, j, t],
                                                        self.y[i, j, k, p, t])
                                                       for (i, j) in self.problem_params.required_edges[t]]))

        # constraint (7)
        for t in range(T):
            for p in range(P[t]):
                for k in range(K):
                    self.model.addConstr(dy_linear[t][p][k] <= self.problem_params.W)

        # constraint (8)
        self.model.addConstrs(self.y[i, j, k, p, t] <= self.x[i, j, k, p, t]
                              for t in range(T)
                              for (i, j) in self.problem_params.required_edges[t]
                              for k in range(K)
                              for p in range(P[t]))

        # constraint (9)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[i, j, k, p, t]
                                                 for (i, j) in self.problem_params.required_edges[t]
                                                 for p in range(P[t])) <= self.problem_params.M * self.u[k, t])

        # constraint (10)
        for t in range(T):
            for p in range(P[t]):
                for k in range(K):
                    self.model.addConstr(self.LT[k, p, t] <= self.problem_params.ul * dy_linear[t][p][k])

        # constraint (11)
        for t in range(T):
            for p in range(P[t]):
                for k in range(K):
                    self.model.addConstr(self.UT[k, p, t] <= self.problem_params.uu * dy_linear[t][p][k])

        # constraint (12)
        for t in range(T):
            for k in range(K):
                LT_UT_xt_linear = gb.LinExpr()
                # add LT to expression
                LT_UT_xt_linear.addTerms([1.0 for _ in range(P[t])],
                                         [self.LT[k, p, t] for p in range(P[t])])
                # add UT to expression
                LT_UT_xt_linear.addTerms([1.0 for _ in range(P[t])],
                                         [self.UT[k, p, t] for p in range(P[t])])
                # add scalar product between x and t to expression
                LT_UT_xt_linear.addTerms([self.problem_params.t[i, j]
                                          for (i, j) in self.problem_params.required_edges[t]
                                          for _ in range(P[t])],
                                         [self.x[i, j, k, p, t]
                                          for (i, j) in self.problem_params.required_edges[t]
                                          for p in range(P[t])])
                # set constraint from expression
                self.model.addConstr(LT_UT_xt_linear <= self.problem_params.T_max)

        # constraint (13)

        # constraint (14)

        # constraint (15)

        # constraint (16)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[1, j, k, 1, t] for j in range(1, V-1)) == self.u[k, t])

        # constraints (17)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[j, V-1, k, 1, t] for j in range(1, V-1)) == self.u[k, t])

        # constraint (18)
        for t in range(T):
            for p in range(1, P[t]):
                for k in range(K):
                    self.model.addConstr(gb.quicksum(self.x[V-1, j, k, p, t] for j in range(1, V-1)) <= self.u[k, t])

        # constraint (19)
        for t in range(T):
            for p in range(1, P[t]):
                for k in range(K):
                    self.model.addConstr(gb.quicksum(self.x[j, V-1, k, p, t] for j in range(1, V-1)) <= self.u[k, t])


class SingleObjectModel0(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        lin_expr = gb.LinExpr()

        theta_c = self.problem_params.theta * self.problem_params.c
        lin_expr.addTerms([theta_c[i, j] for i in range(V)
                           for j in range(V)
                           for t in range(T)
                           for _ in range(P[t])
                           for _ in range(K)],
                          [self.x[i, j, k, p, t] for i in range(V)
                           for j in range(V)
                           for t in range(T)
                           for p in range(P[t])
                           for k in range(K)])

        lin_expr.addTerms([self.problem_params.cv[k] for k in range(K)
                           for _ in range(T)],
                          [self.u[k, t] for k in range(K)
                           for t in range(T)])

        # set objective function


class SingleObjectModel1(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # set objective function


class SingleObjectModel2(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for maximization
        self.model.modelSense = gb.GRB.MAXIMIZE
        
        # set objective function


class SingleObjectModel3(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # set objective function
