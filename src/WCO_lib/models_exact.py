import gurobipy as gb
import numpy as np

from .params import ProblemParams


class BaseModel:
    """
    Base class for setting variables and constraints of the model, and to retrieve
    solutions.
    NOTICE: this is just a "template" class, it should NOT be used directly.
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

        # list for precomputed WT
        self.WT = None

        # flag to check whether the model was solved
        self.model_solved = False

        # solutions to be returned
        self.x_best = None
        self.y_best = None
        self.u_best = None
        self.LT_best = None
        self.UT_best = None

    def set_up_model(self):
        """
        Set constraints common to all single-objective problems and to multi-objective
        one.
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

        # set model constraints
        #
        # (reference numbers are the same as in the original paper)
        # (constraints (20) and (21) are already included inside the definition of the variables)

        # constraint (5)

        # constraint (6)
        for t in range(T):
            for (i, j) in self.problem_params.required_edges[t]:
                self.model.addConstr(gb.quicksum(self.y[i, j, k, p, t] + self.y[j, i, k, p, t]
                                                 for k in range(K)
                                                 for p in range(P[t])) == 1)

        # precompute linear expression for scalar product between d and y
        dy_linear = []
        for t in range(T):
            dy_linear.append([])
            for p in range(P[t]):
                dy_linear[t].append([])
                for k in range(K):
                    dy_linear[t][p].append(gb.LinExpr([(self.problem_params.d[i, j, t], self.y[i, j, k, p, t])
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
        self.WT = []
        for k in range(K):
            self.WT.append([])
            for t in range(T):
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
                # save precomputed expression for object function 3
                self.WT[k].append(LT_UT_xt_linear)

        # constraint (13)
        for t in range(T):
            for p in range(P[t]):
                for k in range(K):
                    self.model.addCons

        # constraint (14)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[1, j, k, 1, t] for j in range(V)) >=
                                     gb.quicksum(self.x[V-1, j, k, 2, t] for j in range(V)))

        # constraint (15)
        for t in range(T):
            for k in range(K):
                # precompute linear expressions
                sum_x_linear = []
                for p in range(1, P[t]):
                    sum_x_linear.append(gb.LinExpr([(1.0, self.x[V-1, j, k, p, t]) for j in range(V)]))
                # set constraints
                for p in range(1, P[t]-1):
                    self.model.addConstr(sum_x_linear[p] >= sum_x_linear[p+1])

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

    def solve(self) -> None:
        """
        Solve the model.
        """
        self.model.optimize()

        # get best solutions
        x_out = self.model.getAttr("x", self.x)
        y_out = self.model.getAttr("x", self.y)
        u_out = self.model.getAttr("x", self.u)
        LT_out = self.model.getAttr("x", self.LT)
        UT_out = self.model.getAttr("x", self.UT)

        # transform output in a more readable format
        #
        # (NOTICE that indexes are inverted in the output)
        self.x_best = [[[[[x_out[i, j, k, p, t] for i in range(self.V)]
                          for j in range(self.V)]
                         for k in range(self.K)]
                        for p in range(self.P[t])]
                       for t in range(self.T)]
        self.y_best = [[[[[y_out[i, j, k, p, t] for i in range(self.V)]
                          for j in range(self.V)]
                         for k in range(self.K)]
                        for p in range(self.P[t])]
                       for t in range(self.T)]
        self.u_best = [[u_out[k, t] for k in range(self.K)] for t in range(self.T)]
        self.LT_best = [[[LT_out[k, p, t] for k in range(self.K)] for p in range(self.P[t])] for t in range(self.T)]
        self.UT_best = [[[UT_out[k, p, t] for k in range(self.K)] for p in range(self.P[t])] for t in range(self.T)]

        # set flag to True for other methods
        self.model_solved = True

    def return_status(self):
        """
        Return the status of the optimization.
        """
        if not self.model_solved:
            print("ERROR: model was not solved. Please run 'model.solve()' before retrieving status.")
            return

        return self.model.status

    def return_objective(self):
        """
        Return the value of the objective function computed on the best solutions.
        """
        if not self.model_solved:
            print("ERROR: model was not solved. Please run 'model.solve()' before retrieving objective.")
            return

        return self.model.objVal

    def return_best_solution(self):
        """
        Return the best solution found.
        """
        if not self.model_solved:
            print("ERROR: model was not solved. Please run 'model.solve()' before retrieving solution.")
            return

        if self.model.status == gb.GRB.OPTIMAL:
            print("WARNING: index order is inverted in the output.")

            return {"x": self.x_best,
                    "y": self.y_best,
                    "u": self.u_best,
                    "LT": self.LT_best,
                    "UT": self.UT_best}

        else:
            print("No optimal solution found: check the status of the optimization for details.")


class SingleObjectModel0(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self) -> None:
        """
        Set constraints and objective of single objective model.
        """
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        obj_function = self.problem_params.theta * gb.LinExpr([(self.problem_params.c[i, j], self.x[i, j, k, p, t])
                                                               for i in range(V)
                                                               for j in range(V)
                                                               for t in range(T)
                                                               for p in range(P[t])
                                                               for k in range(K)])

        obj_function.addTerms([self.problem_params.cv[k] for k in range(K) for _ in range(T)],
                              [self.u[k, t] for k in range(K) for t in range(T)])

        # set objective function
        self.model.setObjective(obj_function)


class SingleObjectModel1(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        """
        Set constraints and objective of single objective model.
        """
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        V = self.V
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        obj_function = gb.LinExpr([(self.problem_params.G[i, j], self.x[i, j, k, p, t])
                                   for i in range(V)
                                   for j in range(V)
                                   for t in range(T)
                                   for p in range(P[t])
                                   for k in range(K)])

        # set objective function
        self.model.setObjective(obj_function)


class SingleObjectModel2(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        """
        Set constraints and objective of single objective model.
        """
        super().set_up_model()

        # rename sets for convenience
        K = self.K
        T = self.T

        # set model for maximization
        self.model.modelSense = gb.GRB.MAXIMIZE

        # build objective function
        obj_function = gb.LinExpr([(self.problem_params.sigma, self.u[k, t])
                                   for t in range(T)
                                   for k in range(K)])

        # set objective function
        self.model.setObjective(obj_function)


class SingleObjectModel3(BaseModel):
    """
    Single-objective model for the problem with object Z_0 (Z_1 in the original
    paper).
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        super().__init__(problem_params)

    def set_up_model(self):
        """
        Set constraints and objective of single objective model.
        """
        super().set_up_model()

        # rename sets for convenience
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        obj_function = gb.LinExpr([(1.0, self.WT[k][t]) for k in range(K) for t in range(T)])

        obj_function = T * K - obj_function / self.problem_params.T_max

        # set objective function
        self.model.setObjective(obj_function)


class SingleObjectModelMain(SingleObjectModel0):
    """
    Single-objective model with objective 0 and epsilon constraints.
    """
    # TODO: just add epsilon constraints to set_up_model
    pass


def compute_objective0(theta: float, c: np.ndarray, cv: np.ndarray, x: list, u: list) -> float:
    """
    Compute the value of the objective function Z_0 (Z_1 in the original
    paper).
    """
    # rename vars for convenience
    T = len(x)
    K = len(x[0][0])
    V = len(x[0][0][0])

    # compute objective
    obj = 0
    for i in range(V):
        for j in range(V):
            partial = 0
            for t in range(T):
                for p in range(len(x[t])):
                    for k in range(K):
                        partial += x[t][p][k][j][i]
            obj += c[i, j] * partial
    obj *= theta

    for k in range(K):
        partial = 0
        for t in range(T):
            partial += u[t][k]
        obj += cv[k] * partial

    return obj


def compute_objective1(G: np.ndarray, x: list) -> float:
    """
    Compute the value of the objective function Z_1 (Z_2 in the original
    paper).
    """
    # rename vars for convenience
    T = len(x)
    K = len(x[0][0])
    V = len(x[0][0][0])

    # compute objective
    obj = 0
    for i in range(V):
        for j in range(V):
            partial = 0
            for t in range(T):
                for p in range(len(x[t])):
                    for k in range(K):
                        partial += x[t][p][k][j][i]
            obj += G[i, j] * partial

    return obj


def compute_objective2(sigma: float, u: list) -> float:
    """
    Compute the value of the objective function Z_2 (Z_3 in the original
    paper).
    """
    return sigma * sum(u)


def compute_objective3(t: np.ndarray, T_max: float, LT: list, UT: list, x: list) -> float:
    """
    Compute the value of the objective function Z_3 (Z_4 in the original
    paper).
    """
    # rename vars for convenience
    T = len(x)
    K = len(x[0][0])
    V = len(x[0][0][0])

    # compute WT
    WT = np.zeros((K, T), dtype=np.float64)

    for k in range(K):
        for t_idx in range(T):
            LT_UT_xt = 0
            for p in range(len(x[t_idx])):
                LT_UT_xt += LT[t_idx][p][k] + UT[t_idx][p][k]
                for i in range(V):
                    for j in range(V):
                        LT_UT_xt += t[i, j] * x[t_idx][p][k][j][i]
            WT[k, t_idx] = LT_UT_xt

    # compute objective
    obj = K * T - float(np.sum(WT)) / T_max

    return obj
