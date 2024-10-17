import numpy as np
import networkx as nx
import copy

from .params import ProblemParams
from .mutations import (edge_swap,
                        trip_shuffle,
                        trip_reverse,
                        trip_combine)


def find_shortest_path(graph: nx.Graph, start: int, end: int) -> list:
    """
    Find the shortest path between two vertices in a graph.
    In particular, return the list of tuples representing the edges of the path from
    the end node of first edge to the start node of the second edge.
    """
    # find shortest path
    node_path = nx.shortest_path(graph, source=start, target=end, weight='weight')

    # Convert the node path to an edge path
    edge_path = []
    for i in range(len(node_path) - 1):
        edge_path.append((node_path[i], node_path[i+1]))

    return edge_path


def compute_service_time(edge_demand: float,
                         edge_traversing_time: float,
                         ul: float,
                         uu: float) -> float:
    """
    Compute the service time of a vehicle for a given edge traversed.
    """
    service_time = edge_demand * (ul + uu) + edge_traversing_time

    return service_time


def update_capacity(current_capacity: float, edge_demand: float) -> float:
    """
    Update the capacity of a vehicle based on last edge traversed.
    """
    new_capacity = current_capacity - edge_demand

    return new_capacity


class SinglePeriodVectorSolution:
    """
    Class to store solution of the problem (for a single period) in vector
    format.
    """

    def __init__(self, period: int):
        self.period = period  # index of the period
        self.first_part = None
        self.second_part = None
        self.service_times = None
        self.total_service_time = None
        self.capacities = None  # remaining capacities of each vehicle
        self.traversals = None  # number of traversals of each edge
        self.total_travelled_distance = None
        self.vehicle_employed = None  # whether each vehicle was employed in this period
        self.objectives = None  # (partial) objective functions

        # only to check feasibility of updated solutions
        self.min_capacities = None

    def __eq__(self, other):
        """
        Check if two solutions are equal.

        Only first and second part are checked.
        """
        if not isinstance(other, SinglePeriodVectorSolution):
            return False

        if np.any(self.first_part != other.first_part):
            return False

        if np.any(self.second_part != other.second_part):
            return False

        return True

    def set_first_part(self, first_part: np.ndarray):
        """
        Set the first part of the solution to a given vector.
        """
        assert len(first_part.shape) == 1, "First part must be a 1D vector."
        if self.second_part is not None:
            assert first_part.shape[0] == self.second_part.shape[0], "First part must have the same size as the second part."

        # set first part of the solution to a given vector
        self.first_part = copy.deepcopy(first_part)

    def set_second_part(self, second_part: np.ndarray):
        """
        Set the second part of the solution to a given vector.
        """
        assert len(second_part.shape) == 1, "Second part must be a 1D vector."
        if self.first_part is not None:
            assert second_part.shape[0] == self.first_part.shape[0], "Second part must have the same size as the first part."

        # set second part of the solution to a given vector
        self.second_part = copy.deepcopy(second_part)

    def adjust_first_part(self, problem_params: ProblemParams):
        """
        Adjust the first part of the solution to satisfy constraints.
        """
        assert False, "This method has not been implemented yet."

        # check coherence
        assert self.second_part is not None, "Second part must be set before the first part is adjusted."

        # compute first part
        # TODO

    def adjust_second_part(self, problem_params: ProblemParams):
        """
        Adjust the second part of the solution to satisfy constraints.
        """
        assert False, "This method has not been implemented yet."

        # check coherence
        assert self.first_part is not None, "First part must be set before the second part is adjusted."

        # compute second part
        # TODO

    def init_heuristic(self, problem_params: ProblemParams):
        """
        Initialize the solution with a heuristic.
        """
        # initialize first and second part
        self.first_part = np.full(problem_params.num_required_edges, -1)
        self.second_part = np.full(problem_params.num_required_edges, -1)

        # initialize auxiliary variables
        service_times = np.zeros(problem_params.num_vehicles)
        capacities = np.full(problem_params.num_vehicles, problem_params.W)
        traversals = np.zeros_like(problem_params.c)  # number of traversals of each edge
        travelled_distance = 0  # total travelled distance
        positions = np.full(problem_params.num_vehicles, 0)  # current position of each vehicle
        available_vehicles = np.full(problem_params.num_vehicles, True)

        # select random vehicle and mark as not available
        current_vehicle = np.random.randint(problem_params.num_vehicles)
        available_vehicles[current_vehicle] = False

        # initialize vehicle employment
        self.vehicle_employed = np.full(problem_params.num_vehicles, False)

        # iterate until all required edges are covered
        d_temp = copy.deepcopy(problem_params.d[:, :, self.period])
        required_edges_temp = copy.deepcopy(problem_params.required_edges)[self.period]
        solution_idx = 0  # index of the current element of the solution
        while required_edges_temp:
            # select current position
            current_position = positions[current_vehicle]

            # find closest starting node of a required edge to current position
            candidate_next_starts = np.where(np.any(d_temp > 0, axis=1))[0]  # possible starts of required edges

            next_start = None
            shortest_path_tonextstart = None
            distance_tonextstart = np.inf
            for candidate in candidate_next_starts:
                candidate_shortest_path = find_shortest_path(problem_params.graph,
                                                             current_position,
                                                             candidate)
                candidate_distance = np.sum([problem_params.c[candidate_shortest_path[j][0],
                                                              candidate_shortest_path[j][1]]
                                             for j in range(len(candidate_shortest_path))])

                if candidate_distance < distance_tonextstart:
                    shortest_path_tonextstart = candidate_shortest_path
                    distance_tonextstart = candidate_distance
                    next_start = candidate

            # choose ending node of required edge at random among non-zero demand edges
            next_end = np.random.choice(np.nonzero(d_temp[next_start])[0])

            # compute service time for possible next position
            next_service_time = np.sum([problem_params.t[shortest_path_tonextstart[j][0],
                                                         shortest_path_tonextstart[j][1]]
                                        for j in range(len(shortest_path_tonextstart))])  # time to go to required edge
            next_service_time += compute_service_time(problem_params.d[next_start, next_end],
                                                      problem_params.t[next_start, next_end],
                                                      problem_params.ul,
                                                      problem_params.uu)  # add time for service of required edge

            shortest_path_todisp = find_shortest_path(problem_params.graph,
                                                      next_end,
                                                      problem_params.num_nodes-1)  # shortest path from end to disposal site
            next_service_time_tot = next_service_time + np.sum([problem_params.t[shortest_path_todisp[j][0],
                                                                                 shortest_path_todisp[j][1]]
                                                                for j in range(len(shortest_path_todisp))])

            # compute remaining capacity for possible next position
            next_capacity = update_capacity(capacities[current_vehicle],
                                            problem_params.d[next_start, next_end])

            # serve next required edge with current vehicle
            if ((service_times[current_vehicle] + next_service_time_tot) < problem_params.T_max
                and next_capacity >= 0):

                # update
                service_times[current_vehicle] += next_service_time
                capacities[current_vehicle] = next_capacity
                positions[current_vehicle] = next_end

                for edge in shortest_path_tonextstart:
                    traversals[edge[0], edge[1]] += 1
                traversals[next_start, next_end] += 1
                travelled_distance += distance_tonextstart
                travelled_distance += problem_params.c[next_start, next_end]

                # update required edges (symmetrically)
                required_edges_temp.remove((next_start, next_end))
                required_edges_temp.remove((next_end, next_start))
                d_temp[next_end, next_start] = 0.
                d_temp[next_start, next_end] = 0.

                # update elements of the solution
                self.first_part[solution_idx] = problem_params.required_edges[self.period].index((next_start, next_end))
                self.second_part[solution_idx] = current_vehicle
                solution_idx += 1

                # mark vehicle as employed
                self.vehicle_employed[current_vehicle] = True

            # go to disposal site
            else:

                # if vehicle already at disposal site, change vehicle (since the current
                # one cannot be employed anymore)
                if current_position == problem_params.num_nodes-1:
                    current_vehicle = np.random.choice(np.where(available_vehicles)[0])
                    available_vehicles[current_vehicle] = False

                # otherwise, move to disposal site and prepare for new trip
                else:
                    # compute shortest path to disposal site
                    shortest_path_todisp = find_shortest_path(problem_params.graph,
                                                              current_position,
                                                              problem_params.num_nodes-1)

                    # update quantities
                    service_times[current_vehicle] += np.sum([problem_params.t[shortest_path_todisp[j][0],
                                                                               shortest_path_todisp[j][1]]
                                                              for j in range(len(shortest_path_todisp))])

                    capacities[current_vehicle] = problem_params.W
                    positions[current_vehicle] = problem_params.num_nodes-1

                    for edge in shortest_path_todisp:
                        traversals[edge[0], edge[1]] += 1
                        travelled_distance += problem_params.c[edge[0], edge[1]]

        # move all remaining vehicles to disposal site
        for vehicle in range(problem_params.num_vehicles):

            if positions[vehicle] != problem_params.num_nodes-1:

                # compute shortest path to disposal site
                shortest_path_todisp = find_shortest_path(problem_params.graph,
                                                          positions[vehicle],
                                                          problem_params.num_nodes-1)

                # update quantities
                service_times[vehicle] += np.sum([problem_params.t[shortest_path_todisp[j][0],
                                                                   shortest_path_todisp[j][1]]
                                                  for j in range(len(shortest_path_todisp))])

                capacities[vehicle] = problem_params.W
                positions[vehicle] = problem_params.num_nodes-1

                for edge in shortest_path_todisp:
                    traversals[edge[0], edge[1]] += 1
                travelled_distance += np.sum([problem_params.c[shortest_path_todisp[j][0],
                                                               shortest_path_todisp[j][1]]
                                              for j in range(len(shortest_path_todisp))])

            # TODO: capire se alla fine si deve tornare al deposito o no

        # save supplementary data
        self.service_times = service_times
        self.total_service_time = np.sum(service_times)
        self.capacities = capacities
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance

        # set minimum capacities to zero (just for coherence)
        self.min_capacities = np.zeros(problem_params.num_vehicles)

    def update_quantities(self, problem_params: ProblemParams) -> None:
        """
        Update the quantities of the solution.
        (To be used when the first and/or second part are changed).
        """
        # reinitialize variables
        service_times = np.zeros(problem_params.num_vehicles)
        capacities = np.full(problem_params.num_vehicles, problem_params.W)
        traversals = np.zeros_like(problem_params.c)
        travelled_distance = 0
        vehicle_employed = np.full(problem_params.num_vehicles, False)
        positions = np.full(problem_params.num_vehicles, 0)

        # initialize minimum capacity
        min_capacities = np.zeros(problem_params.num_vehicles)

        # iterate over required edges in the solution
        for i in range(self.first_part.shape[0]):
            # get required edge and vehicle
            required_edge = problem_params.required_edges[self.period][self.first_part[i]]
            vehicle = self.second_part[i]
            required_start = required_edge[0]
            required_end = required_edge[1]

            # compute shortest path from current position to required edge
            # (we have to choose in what direction to 'take' the edge)
            shortest_path_tostart = find_shortest_path(problem_params.graph,
                                                       positions[vehicle],
                                                       required_start)

            distance_tostart = np.sum([problem_params.c[shortest_path_tostart[j][0],
                                                        shortest_path_tostart[j][1]]
                                       for j in range(len(shortest_path_tostart))])

            # update quantities
            service_times[vehicle] += np.sum([problem_params.t[shortest_path_tostart[j][0],
                                                               shortest_path_tostart[j][1]]
                                              for j in range(len(shortest_path_tostart))])
            service_times[vehicle] += compute_service_time(problem_params.d[required_start,
                                                                            required_end],
                                                           problem_params.t[required_start,
                                                                            required_end],
                                                           problem_params.ul,
                                                           problem_params.uu)

            capacities[vehicle] = update_capacity(capacities[vehicle],
                                                  problem_params.d[required_start, required_end])

            if capacities[vehicle] < min_capacities[vehicle]:
                min_capacities[vehicle] = capacities[vehicle]

            for edge in shortest_path_tostart:
                traversals[edge[0], edge[1]] += 1
            traversals[required_start, required_end] += 1

            travelled_distance += distance_tostart
            travelled_distance += problem_params.c[required_start, required_end]

            vehicle_employed[vehicle] = True

            positions[vehicle] = required_end

            # define next required edge
            next_required_edge = None
            if i != self.first_part.shape[0]-1:
                next_required_edge = problem_params.required_edges[self.period][self.first_part[i+1]]

            # check if vehicle has to go to disposal site
            if (i == self.first_part.shape[0]-1 or
                self.second_part[i+1] != self.second_part[i] or
                problem_params.d[next_required_edge[0], next_required_edge[1]] > capacities[vehicle]):

                # compute shortest path from current position to disposal site
                shortest_path_todisp = find_shortest_path(problem_params.graph,
                                                          positions[vehicle],
                                                          problem_params.num_nodes-1)

                # update quantities
                service_times[vehicle] += np.sum([problem_params.t[shortest_path_todisp[j][0],
                                                                   shortest_path_todisp[j][1]]
                                                  for j in range(len(shortest_path_todisp))])

                capacities[vehicle] = problem_params.W

                for edge in shortest_path_todisp:
                    traversals[edge[0], edge[1]] += 1

                travelled_distance += np.sum([problem_params.c[shortest_path_todisp[j][0],
                                                               shortest_path_todisp[j][1]]
                                              for j in range(len(shortest_path_todisp))])

                positions[vehicle] = problem_params.num_nodes-1

        # save updated variables
        self.service_times = service_times
        self.total_service_time = np.sum(service_times)
        self.capacities = capacities
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance
        self.vehicle_employed = vehicle_employed
        self.min_capacities = min_capacities

    def compute_objectives(self, problem_params: ProblemParams):
        """
        Compute the objective functions of the solution.
        """
        self.objectives = np.zeros(4)

        # compute total waste collection routing cost
        self.objectives[0] = (problem_params.theta * self.total_travelled_distance
                              + problem_params.cv.dot(self.vehicle_employed))

        # compute total pollution routing cost
        self.objectives[1] = np.sum(problem_params.G * self.traversals)

        # compute total amount of hired labor (actually we adjust for minimization)
        self.objectives[2] = problem_params.sigma * (problem_params.num_vehicles
                                                     - np.sum(self.vehicle_employed))

        # compute total work deviation
        self.objectives[3] = problem_params.num_vehicles - self.total_service_time / problem_params.T_max

    def mutate(self):
        """
        Mutate the solution randomly.
        """
        # choose mutation operator
        mutation_type = np.random.choice(["edge_swap",
                                          "trip_shuffle",
                                          "trip_reverse",
                                          "trip_combine"])

        # apply mutation operator
        if mutation_type == "edge_swap":
            edge_swap(self)
        elif mutation_type == "trip_shuffle":
            trip_shuffle(self)
        elif mutation_type == "trip_reverse":
            trip_reverse(self)
        elif mutation_type == "trip_combine":
            trip_combine(self)

    def is_feasible(self, problem_params: ProblemParams) -> bool:
        """
        Check if the solution is feasible for the given problem.
        """
        feasible = True

        if np.any(self.service_times > problem_params.T_max):
            feasible = False

        if np.any(self.min_capacities < 0):
            feasible = False

        return feasible


def generate_heuristic_solution(problem_params: ProblemParams) -> list:
    """
    Generate a (single) initial solution to the problem according to the first
    heuristic in the paper.
    """
    solution_heuristic = []

    # generate solutions for each period with heuristic
    for period in range(problem_params.num_periods):
        period_solution = SinglePeriodVectorSolution(period)
        period_solution.init_heuristic(problem_params)
        solution_heuristic.append(period_solution)

    return solution_heuristic


def dominates(target_objective: np.ndarray, comparison_objective: np.ndarray) -> bool:
    """
    Check if the target solution dominates the comparison solution.
    """
    assert target_objective.shape == comparison_objective.shape, "Objective functions must have the same shape."

    if np.all(target_objective >= comparison_objective):
        if np.any(target_objective > comparison_objective):
            return True

    return False


def geometric_cooling(T: float, alpha: float) -> float:
    """
    Apply geometric cooling to the temperature.
    """
    assert T > 0, "Temperature must be positive."
    assert 0 < alpha < 1, "Alpha must be between 0 and 1."

    T_new = alpha * T

    return T_new


def acceptance_probability(current_objective_functions: np.ndarray,
                           ngbr_objective_functions: np.ndarray,
                           T: float,
                           K: float) -> float:
    """
    Compute the acceptance probability for a non-dominant neighbor solution.
    """
    assert T > 0, "Temperature must be positive."
    assert K > 0, "K must be positive."
    assert current_objective_functions.shape == ngbr_objective_functions.shape, "Objective functions must have the same shape."

    # compute average difference between objective functions
    diff = np.mean(current_objective_functions - ngbr_objective_functions)

    # compute acceptance probability
    prob = np.exp(-diff / (K * T))

    return min(1, prob)


def MOSA(initial_solution: list,
         problem_params: ProblemParams,
         T_0: float = 800.0,
         max_iter: int = 200,
         max_non_improving_iter: int = 10,
         alpha: float = 0.9,
         K: float = 70.0) -> list:
    """
    Apply Multi-Objective Simulated Annealing (MOSA) to a (single) initial solution.
    """
    T = T_0
    current_solution = copy.deepcopy(initial_solution)

    # compute objective functions for the current solution
    current_objective_functions = np.array([0., 0., 0., 0.])
    for period_solution in current_solution:
        period_solution.compute_objectives(problem_params)
        current_objective_functions += period_solution.objectives

    n_iter = 0  # number of iterations
    non_improving_iter = 0  # number of non-improving iterations

    while n_iter < max_iter and non_improving_iter < max_non_improving_iter:
        ngbr_solution = []

        # generate neighbor solution
        for period in range(problem_params.num_periods):

            # get first and second part of the initial period solution
            current_first_part = copy.deepcopy(current_solution[period].first_part)
            current_second_part = copy.deepcopy(current_solution[period].second_part)

            ngbr_solution.append(SinglePeriodVectorSolution(period))

            is_solution_feasible = False  # flag to check if the computed neighbor solution is feasible
            while not is_solution_feasible:

                # generate neighbor period solution
                if np.random.uniform() < 0.5:
                    # perturb first part
                    ngbr_solution[period].set_first_part(np.random.permutation(current_first_part.shape[0]))

                    # copy second part
                    ngbr_solution[period].set_second_part(current_second_part)

                    # update quantities of the neighbor solution
                    ngbr_solution[period].update_quantities(problem_params)

                    # check feasibility
                    is_solution_feasible = ngbr_solution[period].is_feasible(problem_params)

                else:
                    # copy first part
                    ngbr_solution[period].set_first_part(current_first_part)

                    # perturb second part
                    substitute_idx = np.random.randint(current_second_part.shape[0])
                    vehicle_to_substitute = current_second_part[substitute_idx]
                    new_vehicle = np.random.choice(np.delete(np.arange(problem_params.num_vehicles),
                                                             vehicle_to_substitute))
                    ngbr_second_part = copy.deepcopy(current_second_part)
                    ngbr_second_part[substitute_idx] = new_vehicle
                    ngbr_solution[period].set_second_part(ngbr_second_part)

                    # update quantities of the neighbor solution
                    ngbr_solution[period].update_quantities(problem_params)

                    # check feasibility
                    is_solution_feasible = ngbr_solution[period].is_feasible(problem_params)

        # compute objective functions for the neighbor solution
        ngbr_objective_functions = np.array([0., 0., 0., 0.])
        for period_solution in ngbr_solution:
            period_solution.compute_objectives(problem_params)
            ngbr_objective_functions += period_solution.objectives

        # increase number of non-improving iterations
        non_improving_iter += 1

        # accept neighbor solution with probability if dominanted
        if dominates(current_objective_functions, ngbr_objective_functions):
            accept_prob = acceptance_probability(current_objective_functions, ngbr_objective_functions, T, K)
            if accept_prob > np.random.uniform():
                current_solution = ngbr_solution
                current_objective_functions = ngbr_objective_functions
                non_improving_iter = 0

        # accept neighbor solution if dominant or if no dominance occurs
        else:
            current_solution = ngbr_solution
            current_objective_functions = ngbr_objective_functions
            non_improving_iter = 0

        # temperature cooling
        T = geometric_cooling(T, alpha=alpha)

        n_iter += 1

    return current_solution
