import gurobipy as gb

from .params import ProblemParams
from .models_exact import (SingleObjectModel0,
                           SingleObjectModel1,
                           SingleObjectModel2,
                           SingleObjectModel3,
                           SingleObjectModelMain,
                           compute_objective0,
                           compute_objective1,
                           compute_objective2,
                           compute_objective3)


class EpsilonSolver:
    """
    Solver class to solve the optimization problem using epsilon-constraint
    approach.
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        self.problem_params = problem_params
        self.objectives = []

    def solve_single_objectives(self) -> None:
        """
        Solve single-objective problems separately.
        """
        # solve all single-objective models
        model0 = SingleObjectModel0(self.problem_params)
        model0.set_up_model()
        model0.solve()

        model1 = SingleObjectModel1(self.problem_params)
        model1.set_up_model()
        model1.solve()

        model2 = SingleObjectModel2(self.problem_params)
        model2.set_up_model()
        model2.solve()

        model3 = SingleObjectModel3(self.problem_params)
        model3.set_up_model()
        model3.solve()

        # check models' status
        if model0.return_status() == gb.GRB.OPTIMAL:
            print("WARNING: single-objective model with objective 0 was not solved optimally.")

        if model1.return_status() == gb.GRB.OPTIMAL:
            print("WARNING: single-objective model with objective 1 was not solved optimally.")

        if model2.return_status() == gb.GRB.OPTIMAL:
            print("WARNING: single-objective model with objective 2 was not solved optimally.")

        if model3.return_status() == gb.GRB.OPTIMAL:
            print("WARNING: single-objective model with objective 3 was not solved optimally.")

        # get best solutions
        solutions = [model0.return_best_solution(),
                     model1.return_best_solution(),
                     model2.return_best_solution(),
                     model3.return_best_solution()]

        # compute objective functions for all models
        objectives0 = []
        for i in range(4):
            objectives0.append(compute_objective0(self.problem_params.theta,
                                                  self.problem_params.c,
                                                  self.problem_params.cv,
                                                  solutions[i]["x"],
                                                  solutions[i]["u"]))

        objectives1 = []
        for i in range(4):
            objectives1.append(compute_objective1(self.problem_params.G,
                                                  solutions[i]["x"]))

        objectives2 = []
        for i in range(4):
            objectives2.append(compute_objective2(self.problem_params.sigma,
                                                  solutions[i]["u"]))

        objectives3 = []
        for i in range(4):
            objectives3.append(compute_objective3(self.problem_params.t,
                                                  self.problem_params.T_max,
                                                  solutions[i]["LT"],
                                                  solutions[i]["UT"],
                                                  solutions[i]["x"]))

        # save objectives
        self.objectives = [objectives0, objectives1, objectives2, objectives3]

    def compute_epsilon(self) -> None:
        """
        Compute epsilon values for all objectives.
        """
        # TODO
        pass

    def solve(self) -> None:
        """
        Solve main problem with objective 0 as main objective and others as
        epsilon-constraints.
        """
        # TODO
        pass
