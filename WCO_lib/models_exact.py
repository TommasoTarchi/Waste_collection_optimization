import gurobipy as gb
import numpy as np

from .params import ProblemParams
from .subtours import add_subtours_constraint


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
        self.WT = None

        # compute maximum number of trips (assumed to be equal to the number
        # of required edges)
        self.P = self.problem_params.num_required_edges

        # rename sets for convenience
        self.V = self.problem_params.num_nodes
        self.K = self.problem_params.num_vehicles
        self.T = self.problem_params.num_periods

        # flag to check whether the model was solved
        self.model_solved = False

        # solutions to be returned
        self.x_best = None
        self.y_best = None
        self.u_best = None
        self.LT_best = None
        self.UT_best = None
        self.WT_best = None

    def set_up_model(self):
        """
        Set constraints common to all single-objective problems and to multi-objective
        one.
        """
        self.model = gb.Model("Epsilon-constraint")

        # rename sets vars for convenience
        V = self.V
        K = self.K
        P = self.P
        T = self.T

        # set variables (same indexes and names as in the paper)
        self.x = self.model.addVars([(i, j, k, p, t) for t in range(T)
                                     for p in range(P)
                                     for k in range(K)
                                     for (i, j) in self.problem_params.existing_edges],
                                    vtype=gb.GRB.INTEGER,
                                    lb=0,
                                    name="x")

        self.y = self.model.addVars([(i, j, k, p, t) for t in range(T)
                                     for p in range(P)
                                     for k in range(K)
                                     for (i, j) in self.problem_params.required_edges[t]],
                                    vtype=gb.GRB.BINARY,
                                    name="y")

        self.u = self.model.addVars([(k, t) for t in range(T)
                                     for k in range(K)],
                                    vtype=gb.GRB.BINARY,
                                    name="u")

        self.LT = self.model.addVars([(k, p, t) for t in range(T)
                                      for p in range(P)
                                      for k in range(K)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="LT")

        self.UT = self.model.addVars([(k, p, t) for t in range(T)
                                      for p in range(P)
                                      for k in range(K)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="UT")

        self.WT = self.model.addVars([(k, t) for t in range(T)
                                      for k in range(K)],
                                     vtype=gb.GRB.CONTINUOUS,
                                     lb=0.0,
                                     name="WT")

        # set constraint for WT definition
        for k in range(K):
            for t in range(T):
                LT_UT_xt_linear = gb.LinExpr()
                # add LT to expression
                LT_UT_xt_linear.addTerms([1.0 for _ in range(P)],
                                         [self.LT[k, p, t] for p in range(P)])
                # add UT to expression
                LT_UT_xt_linear.addTerms([1.0 for _ in range(P)],
                                         [self.UT[k, p, t] for p in range(P)])
                # add scalar product between x and t to expression
                LT_UT_xt_linear.addTerms([self.problem_params.t[i, j]
                                          for (i, j) in self.problem_params.existing_edges
                                          for _ in range(P)],
                                         [self.x[i, j, k, p, t]
                                          for (i, j) in self.problem_params.existing_edges
                                          for p in range(P)])

                # add constraint
                self.model.addConstr(self.WT[k, t] == LT_UT_xt_linear)

        # set model constraints
        #
        # (reference numbers are the same as in the original paper)
        # (constraint (13) is added after first optimization (only if needed) for efficiency reasons)
        # (constraints (20) and (21) are already included inside the definition of the variables)

        # constraint (5)
        # compute vertices touched by existing edges (without first and last)
        touched_vertices = []
        for edge in self.problem_params.existing_edges:
            if not edge[0] in touched_vertices:
                touched_vertices.append(edge[0])
            if not edge[1] in touched_vertices:
                touched_vertices.append(edge[1])
        if 0 in touched_vertices:
            touched_vertices.remove(0)
        if V-1 in touched_vertices:
            touched_vertices.remove(V-1)

        # add constraint
        for t in range(T):
            for p in range(P):
                for k in range(K):
                    for i in touched_vertices:
                        outgoing_edges = gb.LinExpr([(1.0, self.x[i, j, k, p, t]) for j in touched_vertices
                                                     if (i, j) in self.problem_params.existing_edges])
                        incoming_edges = gb.LinExpr([(1.0, self.x[j, i, k, p, t]) for j in touched_vertices
                                                     if (j, i) in self.problem_params.existing_edges])
                        self.model.addConstr(outgoing_edges == incoming_edges)

        # constraint (6)
        for t in range(T):
            # compute unique required edges
            unique_required_edges = []
            for (i, j) in self.problem_params.required_edges[t]:
                if not (i, j) in unique_required_edges and not (j, i) in unique_required_edges:
                    unique_required_edges.append((i, j))

            for (i, j) in unique_required_edges:
                # compute linear expression
                lin_expr = gb.LinExpr([(1.0, self.y[i, j, k, p, t]) for k in range(K)
                                       for p in range(P)])
                lin_expr.addTerms([1.0 for _ in range(K)
                                   for _ in range(P)],
                                  [self.y[j, i, k, p, t] for k in range(K)
                                   for p in range(P)])

                # add constraint
                self.model.addConstr(lin_expr == 1, name=f"constraint (6) for edge ({i}, {j}) in period {t}")

        # precompute linear expression for scalar product between d and y
        dy_linear = np.empty((K, P, T), dtype=gb.LinExpr)
        for t in range(T):
            for p in range(P):
                for k in range(K):
                    dy_linear[k, p, t] = gb.LinExpr([(self.problem_params.d[i, j, t], self.y[i, j, k, p, t])
                                                     for (i, j) in self.problem_params.required_edges[t]])

        # constraint (7)
        for t in range(T):
            for p in range(P):
                for k in range(K):
                    self.model.addConstr(dy_linear[k, p, t] <= self.problem_params.W)

        # constraint (8)
        self.model.addConstrs(self.y[i, j, k, p, t] <= self.x[i, j, k, p, t]
                              for t in range(T)
                              for p in range(P)
                              for k in range(K)
                              for (i, j) in self.problem_params.required_edges[t])

        # constraint (9)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[i, j, k, p, t] for p in range(P)
                                                 for (i, j) in self.problem_params.required_edges[t])
                                     <= self.problem_params.M * self.u[k, t])

        # constraint (10)
        for t in range(T):
            for p in range(P):
                for k in range(K):
                    self.model.addConstr(self.LT[k, p, t] <= self.problem_params.ul * dy_linear[k, p, t])

        # constraint (11)
        for t in range(T):
            for p in range(P):
                for k in range(K):
                    self.model.addConstr(self.UT[k, p, t] <= self.problem_params.uu * dy_linear[k, p, t])

        # constraint (12)
        self.model.addConstrs(self.WT[k, t] <= self.problem_params.T_max
                              for t in range(T)
                              for k in range(K))

        # constraint (14)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[i, j, k, 0, t] for (i, j) in self.problem_params.existing_edges
                                                 if i == 0) >=
                                     gb.quicksum(self.x[i, j, k, 1, t] for (i, j) in self.problem_params.existing_edges
                                                 if i == V-1),
                                     name=f"constraint (14) for period {t} and vehicle {k}")

        # constraint (15)
        for t in range(T):
            for k in range(K):
                # precompute linear expressions
                sum_x_linear = np.empty(P, dtype=gb.LinExpr)
                for p in range(1, P):
                    sum_x_linear[p] = gb.LinExpr([(1.0, self.x[i, j, k, p, t])
                                                  for (i, j) in self.problem_params.existing_edges
                                                  if i == V-1])
                # set constraints
                for p in range(1, P-1):
                    self.model.addConstr(sum_x_linear[p] >= sum_x_linear[p+1],
                                         name=f"constraint (15) in period {t} and vehicle {k} for trip {p}")

        # constraint (16)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[i, j, k, 0, t] for (i, j) in self.problem_params.existing_edges
                                                 if i == 0 and j in range(1, V-1)) == self.u[k, t])

        # constraints (17)
        for t in range(T):
            for k in range(K):
                self.model.addConstr(gb.quicksum(self.x[i, j, k, 0, t] for (i, j) in self.problem_params.existing_edges
                                                 if i in range(1, V-1) and j == V-1) == self.u[k, t])

        # constraint (18)
        for t in range(T):
            for p in range(1, P):
                for k in range(K):
                    self.model.addConstr(gb.quicksum(self.x[i, j, k, p, t] for (i, j) in self.problem_params.existing_edges
                                                     if i == V-1 and j in range(1, V-1)) <= self.u[k, t])

        # constraint (19)
        for t in range(T):
            for p in range(1, P):
                for k in range(K):
                    self.model.addConstr(gb.quicksum(self.x[i, j, k, p, t] for (i, j) in self.problem_params.existing_edges
                                                     if i in range(1, V-1) and j == V-1) <= self.u[k, t])

    # TODO: add thresholds for subtours constraint as parameters
    def solve(self) -> None:
        """
        Solve the model.
        """
        # solve model a first time
        self.model.optimize()

        # conditionally add subtours constraint
        add_constr = add_subtours_constraint(self.model,
                                             self.x,
                                             self.problem_params.num_nodes,
                                             self.problem_params.existing_edges,
                                             self.K,
                                             self.P,
                                             self.T,
                                             self.problem_params.M,
                                             0.7,
                                             6)

        # if needed, optimize again with new constraint
        if add_constr:
            self.model.optimize()

        # get best solutions
        x_out = self.model.getAttr("x", self.x)
        y_out = self.model.getAttr("x", self.y)
        u_out = self.model.getAttr("x", self.u)
        LT_out = self.model.getAttr("x", self.LT)
        UT_out = self.model.getAttr("x", self.UT)
        WT_out = self.model.getAttr("x", self.WT)

        # transform output in numpy array format
        self.x_best = np.empty((self.K, self.P, self.T, self.problem_params.num_edges * 2), dtype=np.int64)
        self.y_best = np.empty((self.K, self.P, self.T, self.problem_params.num_required_edges * 2), dtype=np.int64)

        for t in range(self.T):
            for p in range(self.P):
                for k in range(self.K):
                    self.x_best[k, p, t, :] = np.array([x_out[i, j, k, p, t] for (i, j) in self.problem_params.existing_edges])
                    self.y_best[k, p, t, :] = np.array([y_out[i, j, k, p, t] for (i, j) in self.problem_params.required_edges[t]])

        self.u_best = np.array([[u_out[k, t] for t in range(self.T)] for k in range(self.K)])
        self.LT_best = np.array([[[LT_out[k, p, t] for t in range(self.T)] for p in range(self.P)] for k in range(self.K)])
        self.UT_best = np.array([[[UT_out[k, p, t] for t in range(self.T)] for p in range(self.P)] for k in range(self.K)])
        self.WT_best = np.array([[WT_out[k, t] for t in range(self.T)] for k in range(self.K)])

        # set flag to True for other methods
        self.model_solved = True

    def return_status(self):
        """
        Return the status of the optimization.
        """
        if not self.model_solved:
            raise Exception("ERROR: model was not solved. Please run 'model.solve()' before retrieving status.")

        return self.model.status

    def return_objective(self):
        """
        Return the value of the objective function computed on the best solutions.
        """
        if not self.model_solved:
            raise Exception("ERROR: model was not solved. Please run 'model.solve()' before retrieving objective.")

        return self.model.objVal

    def return_best_solution(self):
        """
        Return the best solution found.
        """
        if not self.model_solved:
            raise Exception("ERROR: model was not solved. Please run 'model.solve()' before retrieving solution.")

        if self.model.status == gb.GRB.OPTIMAL:
            return {"x": self.x_best,
                    "y": self.y_best,
                    "u": self.u_best,
                    "LT": self.LT_best,
                    "UT": self.UT_best,
                    "WT": self.WT_best}

        else:
            print("No optimal solution found: check the status of the optimization for details.")

    # TODO: togliere quando finito debugging (togliere anche nome da
    #       constraint per cui si sono usate le slack variable)
    def return_slack(self):
        for t in range(self.T):
            unique_required_edges = []
            for (i, j) in self.problem_params.required_edges[t]:
                if not (i, j) in unique_required_edges and not (j, i) in unique_required_edges:
                    unique_required_edges.append((i, j))

            for (i, j) in unique_required_edges:
                constr = self.model.getConstrByName(f"constraint (6) for edge ({i}, {j}) in period {t}")
                slack = constr.slack
                print(f"Constraint {constr.ConstrName} has slack: {slack}")

        for t in range(self.T):
            for k in range(self.K):
                constr = self.model.getConstrByName(f"constraint (14) for period {t} and vehicle {k}")
                slack = constr.slack
                print(f"Constraint {constr.ConstrName} has slack: {slack}")

                for p in range(1, self.P-1):
                    constr = self.model.getConstrByName(f"constraint (15) in period {t} and vehicle {k} for trip {p}")
                    slack = constr.slack
                    print(f"Constraint {constr.ConstrName} has slack: {slack}")


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
        K = self.K
        P = self.P
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        obj_function = self.problem_params.theta * gb.LinExpr([(self.problem_params.c[i, j], self.x[i, j, k, p, t])
                                                               for t in range(T)
                                                               for p in range(P)
                                                               for k in range(K)
                                                               for (i, j) in self.problem_params.existing_edges])

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
        K = self.K
        T = self.T

        # set model for minimization
        self.model.modelSense = gb.GRB.MINIMIZE

        # build objective function
        obj_function = gb.LinExpr([(self.problem_params.G[i, j], self.x[i, j, k, p, t])
                                   for t in range(T)
                                   for p in range(P)
                                   for k in range(K)
                                   for (i, j) in self.problem_params.existing_edges])

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
        obj_function = gb.LinExpr([(1.0, self.WT[k, t]) for k in range(K) for t in range(T)])

        obj_function = T * K - obj_function / self.problem_params.T_max

        # set objective function
        self.model.setObjective(obj_function)


class SingleObjectModelMain(SingleObjectModel0):
    """
    Single-objective model with objective 0 and epsilon constraints.
    """

    def __init__(self, problem_params: ProblemParams,
                 eps1: float,
                 eps2: float,
                 eps3: float) -> None:
        super().__init__(problem_params)

        self.eps1 = eps1
        self.eps2 = eps2
        self.eps3 = eps3

    def set_up_model(self) -> None:
        """
        Set constraints and objective of main single objective model.
        """
        super().set_up_model()

        # rename sets for convenience
        P = self.P
        K = self.K
        T = self.T

        # add epsilon constraints
        self.model.addConstr(gb.LinExpr([(self.problem_params.G[i, j], self.x[i, j, k, p, t])
                                         for t in range(T)
                                         for p in range(P)
                                         for k in range(K)
                                         for (i, j) in self.problem_params.existing_edges]) <= self.eps1)

        self.model.addConstr(gb.LinExpr([(self.problem_params.sigma, self.u[k, t])
                                         for t in range(T)
                                         for k in range(K)]) >= self.eps2)

        lin_expr_part = gb.LinExpr([(1.0, self.WT[k, t]) for k in range(K) for t in range(T)])
        lin_expr = T * K - lin_expr_part / self.problem_params.T_max
        self.model.addConstr(lin_expr <= self.eps3)


def compute_objective0(theta: float,
                       c: np.ndarray,
                       cv: np.ndarray,
                       existing_edges: list,
                       x: np.ndarray,
                       u: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_0 (Z_1 in the original paper).
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
    """
    obj = 0
    partial_sums = np.sum(x, axis=(0, 1, 2))
    for (count, (i, j)) in enumerate(existing_edges):
        obj += G[i, j] * partial_sums[count]

    return obj


def compute_objective2(sigma: float, u: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_2 (Z_3 in the original paper).
    """
    return sigma * np.sum(u)


def compute_objective3(T_max: float,
                       num_vehicles: int,
                       num_periods: int,
                       WT: np.ndarray) -> float:
    """
    Compute the value of the objective function Z_3 (Z_4 in the original paper).
    """
    return num_vehicles * num_periods - float(np.sum(WT)) / T_max
