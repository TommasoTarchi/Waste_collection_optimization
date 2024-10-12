import numpy as np
import networkx as nx

from .params import ProblemParams


def find_shortest_path(graph: nx.Graph, start_edge: tuple, end_edge: tuple) -> list:
    """
    Find the shortest path between two edges in a graph.
    In particular, return the list of tuples representing the edges of the path from
    the end node of first edge to the start node of the second edge.
    """
    # extract first and last nodes of the path
    source = start_edge[1]
    target = end_edge[0]

    # find shorted path
    node_path = nx.shortest_path(graph, source=source, target=target)

    # Convert the node path to an edge path
    edge_path = []
    for i in range(len(node_path) - 1):
        edge_path.append((node_path[i], node_path[i+1]))

    return edge_path


class SinglePeriodVectorSolution:
    """
    Class to store solution of the problem (for a single period) in vector
    format.
    """

    def __init__(self, period: int):
        self.period = period  # index of the period
        self.first_part = None
        self.second_part = None
        self.total_service_time = None
        self.capacieties = None  # remaining capacities of each vehicle
        self.traversals = None  # number of traversals of each edge
        self.total_travelled_distance = None
        self.vehicle_employed = None  # whether each vehicle was employed in this period
        self.objectives = None  # (partial) objective functions

    def set_first_part(self, first_part: np.ndarray):
        """
        Set the first part of the solution to a given vector.
        """
        assert len(first_part.shape) == 1, "First part must be a 1D vector."
        if self.second_part is not None:
            assert first_part.shape[0] == self.second_part.shape[0], "First part must have the same size as the second part."

        # set first part of the solution to a given vector
        self.first_part = first_part

    def set_second_part(self, second_part: np.ndarray):
        """
        Set the second part of the solution to a given vector.
        """
        assert len(second_part.shape) == 1, "Second part must be a 1D vector."
        if self.first_part is not None:
            assert second_part.shape[0] == self.first_part.shape[0], "Second part must have the same size as the first part."

        # set second part of the solution to a given vector
        self.second_part = second_part

    def adjust_first_part(self, problem_params: ProblemParams):
        """
        Adjust the first part of the solution to satisfy constraints.
        """
        # check coherence
        assert self.second_part is not None, "Second part must be set before the first part is adjusted."

        # compute first part
        # TODO

    def adjust_second_part(self, problem_params: ProblemParams):
        """
        Adjust the second part of the solution to satisfy constraints.
        """
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
        d_temp = problem_params.d[:, :, self.period].copy()
        required_edges_temp = problem_params.required_edges.copy()
        solution_idx = 0  # index of the current element of the solution
        while required_edges_temp:
            # select current position
            current_position = positions[current_vehicle]

            # find closest starting node of a required edge to current position
            candidate_next_starts = np.where(np.any(d_temp > 0, axis=1))[0]
            next_start = candidate_next_starts[np.argmin(problem_params.c[current_position, candidate_next_starts])]

            # choose ending node of required edge at random among non-zero demand edges
            next_end = np.random.choice(np.nonzero(next_start)[0])

            # compute service time for possible next position
            next_service_time = problem_params.c[current_position, next_start]  # time to go to required edge
            next_service_time += compute_service_time(problem_params.d[next_start, next_end],
                                                      problem_params.t[next_start, next_end],
                                                      problem_params.ul,
                                                      problem_params.uu)  # add time for service of required edge
            next_service_time_tot = problem_params.t[next_end, problem_params.num_nodes-1]  # add time to go to disposal site
            next_service_time_tot += problem_params.t[problem_params.num_nodes-1, 0]  # add time to go back to depot

            # compute remaining capacity for possible next position
            next_capacity = update_capacity(capacities[current_vehicle], problem_params.d[next_start, next_end])

            # serve next required edge with current vehicle
            if next_service_time_tot < problem_params.T_max and next_capacity >= 0:
                # update
                service_times[current_vehicle] = next_service_time
                capacities[current_vehicle] = next_capacity
                d_temp[next_start, next_end] = 0
                positions[current_vehicle] = next_end
                traversals[next_start, next_end] += 1
                travelled_distance += problem_params.c[current_position, next_start] + problem_params.c[next_start, next_end]

                # update required edges (symmetrically)
                if next_end < next_start:
                    next_start, next_end = next_end, next_start
                required_edges_temp.remove((next_start, next_end))

                # update elements of the solution
                self.first_part[solution_idx] = problem_params.required_edges.index((next_start, next_end))
                self.second_part[solution_idx] = current_vehicle
                solution_idx += 1

                # mark vehicle as employed
                self.vehicle_employed[current_vehicle] = True

            # go to disposal site
            elif current_position != problem_params.num_nodes-1:
                # update
                service_times[current_vehicle] = problem_params.t[current_position, problem_params.num_nodes-1]
                capacities[current_vehicle] = problem_params.W
                positions[current_vehicle] = problem_params.num_nodes-1
                traversals[current_position, problem_params.num_nodes-1] += 1
                travelled_distance += problem_params.t[current_position, problem_params.num_nodes-1]

            # go back to depot
            else:
                # update
                service_times[current_vehicle] = problem_params.t[problem_params.num_nodes-1, 0]
                positions[current_vehicle] = 0
                traversals[problem_params.num_nodes-1, 0] += 1
                travelled_distance += problem_params.t[problem_params.num_nodes-1, 0]

                # select new vehicle among the available ones
                current_vehicle = np.random.choice(np.where(available_vehicles)[0])
                available_vehicles[current_vehicle] = False

        # move all vehicles to depot and update service times
        for vehicle in range(problem_params.num_vehicles):
            if positions[vehicle] != 0:
                service_times[vehicle] += problem_params.t[positions[vehicle], problem_params.num_nodes-1] + problem_params.t[problem_params.num_nodes-1, 0]

        # save supplementary data
        self.total_service_time = np.sum(service_times)
        self.capacieties = capacities
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance

    def update_quantities(self, problem_params: ProblemParams):
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

        # build networkx graph
        G = nx.Graph()
        G.add_edges_from(problem_params.existing_edges[self.period])

        # iterate over elements of the solution
        for i in range(self.first_part.shape[0]):
            # TODO
            pass

        # update variables
        self.total_service_time = np.sum(service_times)
        self.capacities = capacities
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance
        self.vehicle_employed = vehicle_employed

    def compute_objectives(self, problem_params: ProblemParams):
        """
        Compute the objective functions of the solution.
        """
        self.objectives = np.zeros(4)

        # compute total waste collection routing cost
        self.objectives[0] = problem_params.theta * self.total_travelled_distance + problem_params.cv.dot(self.vehicle_employed)

        # compute total pollution routing cost
        self.objectives[1] = np.sum(problem_params.G * self.traversals)

        # compute total amount of hired labor (actually we change sign for minimization)
        self.objectives[2] = -problem_params.sigma * np.sum(self.vehicle_employed)

        # compute total work deviation
        self.objectives[3] = np.sum(1 - self.total_service_time / problem_params.T_max)

    def mutate(self, problem_params: ProblemParams):
        """
        Mutate the solution.
        """
        # TODO

    def is_feasible(self, problem_params: ProblemParams) -> bool:
        """
        Check if the solution is feasible for the given problem.
        """
        feasible = True

        if self.total_service_time > problem_params.T_max:
            feasible = False

        if np.any(self.capacieties < 0):
            feasible = False

        return feasible


def generate_heuristic_solution(problem_params: ProblemParams) -> list:
    """
    Generate a (single) initial solution to the problem according to the first heuristic
    in the paper.
    """
    solution_heuristic = []

    # generate solutions for each period with heuristic
    for period in range(problem_params.num_periods):
        solution = SinglePeriodVectorSolution(period)
        solution.init_heuristic(problem_params)
        solution_heuristic.append(solution)

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
    current_solution = initial_solution

    # compute objective functions for the current solution
    current_objective_functions = np.array([0, 0, 0, 0])
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
            current_first_part = current_solution[period].first_part.copy()
            current_second_part = current_solution[period].second_part.copy()

            ngbr_solution.append(SinglePeriodVectorSolution(period))

            # generate neighbor period solution
            if np.random.uniform() < 0.5:
                # perturb first part
                ngbr_solution[period].set_first_part(np.random.permutation(current_first_part.size[0]))

                # copy second part
                ngbr_solution[period].set_second_part(current_second_part)

                # adjust second part to satisfy constraints
                ngbr_solution[period].adjust_second_part(problem_params)

            else:
                # copy first part
                ngbr_solution[period].set_first_part(current_first_part)

                # perturb second part
                vehicle_to_substitute = np.random.choice(current_second_part)
                new_vehicle = np.random.randint(problem_params.num_vehicles)
                ngbr_second_part = current_second_part.copy()
                ngbr_second_part[ngbr_second_part == vehicle_to_substitute] = new_vehicle
                ngbr_solution[period].set_second_part(ngbr_second_part)

                # adjust first part to satisfy constraints
                ngbr_solution[period].adjust_first_part(problem_params)

        # compute objective functions for the neighbor solution
        ngbr_objective_functions = np.array([0, 0, 0, 0])
        for period_solution in ngbr_solution:
            period_solution.compute_objectives(problem_params)
            ngbr_objective_functions += period_solution.objectives

        # increase number of non-improving iterations
        non_improving_iter += 1

        # accept neighbor solution if dominant
        if dominates(ngbr_objective_functions, current_objective_functions):
            current_solution = ngbr_solution
            current_objective_functions = ngbr_objective_functions
            non_improving_iter = 0

        # accept neighbor solution with probability if non-dominant
        elif dominates(current_objective_functions, ngbr_objective_functions):
            accept_prob = acceptance_probability(current_objective_functions, ngbr_objective_functions, T, K)
            if accept_prob > np.random.uniform():
                current_solution = ngbr_solution
                current_objective_functions = ngbr_objective_functions
                non_improving_iter = 0

        # temperature cooling
        T = geometric_cooling(T, alpha=alpha)

        n_iter += 1

    return current_solution
