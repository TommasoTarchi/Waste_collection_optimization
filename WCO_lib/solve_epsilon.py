import numpy as np
import gurobipy as gb
from itertools import product
from typing import Optional, List
import copy
import time

from .params import ProblemParams
from .models_exact import (SingleObjectModel0,
                           SingleObjectModel1,
                           SingleObjectModel2,
                           SingleObjectModel3,
                           SingleObjectModelMain)
from .evaluate import (compute_objective0,
                       compute_objective1,
                       compute_objective2,
                       compute_objective3,
                       sort_solutions)


class EpsilonSolver:
    """
    Solver class to solve the optimization problem using epsilon-constraint
    approach.
    """

    def __init__(self, problem_params: ProblemParams) -> None:
        self.problem_params = problem_params
        self.objectives = None
        self.epsilon_values = None
        self.pareto_solutions = None
        self.model_status = None

    def solve_single_objectives(self, time_limit: Optional[float] = None ) -> bool:
        """
        Solve single-objective problems separately.

        A time limit (in seconds) can be passed.
        """
        start_time = time.perf_counter()

        # solve all single-objective models
        model0 = SingleObjectModel0(self.problem_params)
        model0.set_up_model()
        model0.solve()

        elapsed_time = time.perf_counter() - start_time
        if time_limit is not None and elapsed_time > time_limit:
            return True

        model1 = SingleObjectModel1(self.problem_params)
        model1.set_up_model()
        model1.solve()

        elapsed_time = time.perf_counter() - start_time
        if time_limit is not None and elapsed_time > time_limit:
            return True

        model2 = SingleObjectModel2(self.problem_params)
        model2.set_up_model()
        model2.solve()

        elapsed_time = time.perf_counter() - start_time
        if time_limit is not None and elapsed_time > time_limit:
            return True

        model3 = SingleObjectModel3(self.problem_params)
        model3.set_up_model()
        model3.solve()

        elapsed_time = time.perf_counter() - start_time
        if time_limit is not None and elapsed_time > time_limit:
            return True

        # check models' status
        if model0.return_status() == "2":
            print("\nWARNING: single-objective model with objective 0 was not solved optimally.\n")

        if model1.return_status() == "2":
            print("\nWARNING: single-objective model with objective 1 was not solved optimally.\n")

        if model2.return_status() == "2":
            print("\nWARNING: single-objective model with objective 2 was not solved optimally.\n")

        if model3.return_status() == "2":
            print("\nWARNING: single-objective model with objective 3 was not solved optimally.\n")

        # get best solutions
        solutions = [model0.return_best_solution(),
                     model1.return_best_solution(),
                     model2.return_best_solution(),
                     model3.return_best_solution()]

        # compute objective functions for all models (the one corresponding
        # the model optimizing the objective is excluded)
        objectives0 = []
        for i in range(4):
            objectives0.append(compute_objective0(self.problem_params.theta,
                                                  self.problem_params.c,
                                                  self.problem_params.cv,
                                                  self.problem_params.existing_edges,
                                                  solutions[i]["x"],
                                                  solutions[i]["u"]))

        objectives1 = []
        for i in range(4):
            objectives1.append(compute_objective1(self.problem_params.G,
                                                  self.problem_params.existing_edges,
                                                  solutions[i]["x"]))

        objectives2 = []
        for i in range(4):
            objectives2.append(compute_objective2(self.problem_params.sigma,
                                                  solutions[i]["u"]))

        objectives3 = []
        for i in range(4):
            objectives3.append(compute_objective3(self.problem_params.T_max,
                                                  self.problem_params.num_vehicles,
                                                  self.problem_params.num_periods,
                                                  solutions[i]["WT"]))

        # save computed objectives
        self.objectives = [objectives0, objectives1, objectives2, objectives3]

        return False

    def compute_epsilon(self, num_epsilon: int = 4) -> None:
        """
        Compute epsilon values for all objectives.
        """
        # check epsilon positive
        if num_epsilon <= 0:
            raise ValueError("Parameter 'num_epsilon' must be positive")

        if self.objectives is None:
            raise ValueError("Objectives must be computed first. Please run 'solve_single_objectives' method first.")

        epsilons = []
        for i in range(1, 4):
            # remove the objective corresponding to the model
            objectives_copy = copy.deepcopy(self.objectives[i])
            objectives_copy.pop(i)

            # sort objectives
            ordered_objectives = np.unique(objectives_copy)

            # select bounds for epsilon values
            epsilon_inf = None
            epsilon_sup = None

            if len(ordered_objectives) == 1:
                epsilon_inf = ordered_objectives[0]
                epsilon_sup = ordered_objectives[0]
            else:
                if i == 2:
                    epsilon_inf = ordered_objectives[-2]
                    epsilon_sup = ordered_objectives[-1]
                else:
                    epsilon_inf = ordered_objectives[0]
                    epsilon_sup = ordered_objectives[1]

            # compute epsilon values
            epsilons.append(np.unique(np.linspace(epsilon_inf, epsilon_sup, num=num_epsilon)))

        # combine epsilon values
        self.epsilon_values = list(product(*epsilons))

    def solve_multi_objective(self, time_limit: Optional[float] = None) -> bool:
        """
        Solve main problem with objective 0 as main objective and others as
        epsilon-constraints.

        A time limit (in seconds) can be passed.
        """
        if self.objectives is None:
            raise ValueError("Objectives must be computed first. Please run 'solve_single_objectives' method first.")

        if self.epsilon_values is None:
            raise ValueError("Epsilon values must be computed first. Please run 'compute_epsilon' method first.")

        time_limit_exceeded = False
        start_time = time.perf_counter()

        # solve main model for all combinations of epsilon values
        pareto_solutions = []
        model_status = []
        for epsilons in self.epsilon_values:
            model = SingleObjectModelMain(self.problem_params,
                                          epsilons[0],
                                          epsilons[1],
                                          epsilons[2])
            model.set_up_model()
            model.solve()
            if model.return_status() == gb.GRB.OPTIMAL:
                pareto_solutions.append(model.return_best_solution())
                model_status.append(model.return_status())
            #model.return_slack()  # FOR DEBUGGING

            # check time limit
            elapsed_time = time.perf_counter() - start_time
            if time_limit is not None and elapsed_time > time_limit:
                time_limit_exceeded = True
                break

        # remove duplicate solutions
        pareto_solutions_unique = []
        for solution in pareto_solutions:

            add = True
            for solution_unique in pareto_solutions_unique:
                if np.all(solution["x"] == solution_unique["x"]) and np.all(solution["y"] == solution_unique["y"]):
                    add = False
                    break

            if add:
                pareto_solutions_unique.append(solution)

        # compute objectives for all unique Pareto solutions
        pareto_objectives = []
        for solution in pareto_solutions_unique:
            obj0 = compute_objective0(self.problem_params.theta,
                                      self.problem_params.c,
                                      self.problem_params.cv,
                                      self.problem_params.existing_edges,
                                      solution["x"],
                                      solution["u"])
            obj1 = compute_objective1(self.problem_params.G,
                                      self.problem_params.existing_edges,
                                      solution["x"])
            obj2 = (self.problem_params.sigma
                    * self.problem_params.num_vehicles
                    * self.problem_params.num_periods
                    - compute_objective2(self.problem_params.sigma, solution["u"]))
            obj3 = compute_objective3(self.problem_params.T_max,
                                      self.problem_params.num_vehicles,
                                      self.problem_params.num_periods,
                                      solution["WT"])

            pareto_objectives.append([obj0, obj1, obj2, obj3])

        # retain only first Pareto front solutions
        _, first_front_indices = sort_solutions(pareto_objectives)

        self.pareto_solutions = [pareto_solutions_unique[i] for i in first_front_indices]

        # save status of the model
        self.model_status = model_status

        return time_limit_exceeded

    def return_pareto_solutions(self) -> List:
        """
        Return pareto solutions.
        """
        if self.pareto_solutions is None:
            raise ValueError("Pareto solutions must be computed first. Please run 'solve_multi_objective' method first.")

        return self.pareto_solutions

    def return_status(self) -> List:
        """
        Return status of the final models.
        """
        if self.model_status is None:
            raise ValueError("Status of the model must be computed first. Please run 'solve_multi_objective' method first.")

        return self.model_status
