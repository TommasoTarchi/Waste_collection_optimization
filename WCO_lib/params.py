import json
import numpy as np


def check_ProblemParams(params):
    """
    Check if the problem parameters are coherent.
    """
    # check the size of arrays
    assert params.num_edges == int(len(params.existing_edges) / 2), "Number of existing edges not coherent with problem size."
    assert params.c.shape == (params.num_nodes, params.num_nodes), "Shape of c not coherent with problem size."
    assert params.d.shape == (params.num_nodes, params.num_nodes, params.num_periods), "Shape of d not coherent with problem size."
    assert params.t.shape == (params.num_nodes, params.num_nodes), "Shape of t not coherent with problem size."
    assert params.cv.shape == (params.num_vehicles,), "Shape of cv not coherent with problem size."
    assert params.G.shape == (params.num_nodes, params.num_nodes), "Shape of G not coherent with problem size."

    # check demand is non-negative
    assert np.all(params.d >= 0), "Demand d must be non-negative."

    # check that the number of required edges is the same across all periods
    num_req_edge_comp = params.num_required_edges * 2
    for t in range(params.num_periods):
        assert len(params.required_edges[t]) == num_req_edge_comp, "Number of required edges must be coherent with problem size AND the same across all periods."

    # check that required edges exist
    for t in range(params.num_periods):
        for (i, j) in params.required_edges[t]:
            assert (i, j) in params.existing_edges, "Required edges must exist."


def check_MosaMoiwoaSolverParams(params):
    """
    Check if the solver parameters are coherent.
    """
    # check ranges of parameters
    assert params.N_0 > 0, "Initial number of solutions must be positive."
    assert params.MOSA_T_0 > 0, "Initial temperature must be positive."
    assert params.MOSA_max_iter > 0, "Maximum number of iterations for MOSA must be positive."
    assert params.MOSA_max_non_improving_iter > 0, "Maximum number of non-improving iterations for MOSA must be positive."
    assert 0 < params.MOSA_alpha < 1, "Cooling factor must be between 0 and 1."
    assert params.MOSA_K > 0, "Boltzman constant must be positive."
    assert params.MOIWOA_S_min > 0, "Minimum number of children seeds must be positive."
    assert params.MOIWOA_S_max > 0, "Maximum number of children seeds must be positive."
    assert params.MOIWOA_N_max > 0, "Maximum number of solutions must be positive."
    assert params.MOIWOA_max_iter > 0, "Maximum number of iterations for MOIWOA must be positive."

    # check S_min is less than S_max
    assert params.MOIWOA_S_min < params.MOIWOA_S_max, "Minimum number of children seeds must be less than maximum number."


def compute_service_time(edge_demand, edge_traversing_time, ul, uu):
    """
    Compute the service time of a vehicle for a given edge traversed.
    """
    service_time = edge_demand * (ul + uu) + edge_traversing_time

    return service_time


def update_capacity(current_capacity, edge_demand):
    """
    Update the capacity of a vehicle based on last edge traversed.
    """
    new_capacity = current_capacity - edge_demand

    return new_capacity


class ProblemParams:
    """
    Class to store parameters of the problem.
    """

    def __init__(self):
        # size of the problem
        self.num_nodes = 0  # number of nodes
        self.num_edges = 0  # number of existing edges
        self.num_required_edges = 0  # number of required edges per period (counted once in symmetric matrix)
        self.num_vehicles = 0  # number of vehicles
        self.num_periods = 0  # number of planning periods

        # parameters of the problem
        self.c = None  # edge distance (symmetric; elements are zero if edge does not exist)
        self.W = 0  # vehicle capacity
        self.d = None  # edge demand at each period (0 for no demand)
        self.T_max = 0  # maximum available time for vehicles
        self.M = 0  # a large number
        self.t = None  # traversing time of edges
        self.cv = None  # usage cost of vehicles
        self.theta = 0  # conversion factor of distance to cost
        self.G = None  # pollution emitted by traversing edges
        self.sigma = 0  # number of workforce per vehicle
        self.ul = 0  # conversion factor of demand to loading time
        self.uu = 0  # conversion factor of demand to unloading time

        # existing edges coordinates (i is staring point and j is ending point)
        self.existing_edges = None

        # required edges coordinates (i is staring point and j is ending point)
        self.required_edges = None

        # number of periods each vehicle is employed for
        self.periods_employed = None

    def load_from_dir(self, data_dir):
        # define paths to init data
        params_path = data_dir + '/problem_parameters.json'
        c_path = data_dir + '/c.npy'
        d_path = data_dir + '/d.npy'
        t_path = data_dir + '/t.npy'
        cv_path = data_dir + '/cv.npy'
        G_path = data_dir + '/G.npy'

        # load scalar parameters from json
        with open(params_path) as f:
            params = json.load(f)
            self.num_nodes = params['num_nodes']
            self.num_edges = params['num_edges']
            self.num_required_edges = params['num_required_edges']
            self.num_vehicles = params['num_vehicles']
            self.num_periods = params['num_periods']
            self.W = params['W']
            self.T_max = params['T_max']
            self.M = params['M']
            self.theta = params['theta']
            self.sigma = params['sigma']
            self.ul = params['ul']
            self.uu = params['uu']

        # load array parameters from binary files
        self.c = np.load(c_path)
        self.d = np.load(d_path)
        self.t = np.load(t_path)
        self.cv = np.load(cv_path)
        self.G = np.load(G_path)

        # compute existing edges coordinates (list of tuples)
        existing_coord = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.c[i, j] > 0:
                    existing_coord.append((i, j))
                    existing_coord.append((j, i))
        self.existing_edges = existing_coord

        # compute required edges coordinates for each period (list of
        # lists of tuples)
        required_coord = []
        for t in range(self.num_periods):
            required_coord.append([])
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.d[i, j, t] > 0:
                        required_coord[-1].append((i, j))
                        required_coord[-1].append((j, i))
        self.required_edges = required_coord

        # init periods periods employed
        self.periods_employed = np.zeros(self.num_vehicles)

        # check coherence of parameters
        check_ProblemParams(self)

    def save_to_dir(self, data_dir):
        # define paths to init data
        params_path = data_dir + '/problem_parameters.json'
        c_path = data_dir + '/c.npy'
        d_path = data_dir + '/d.npy'
        t_path = data_dir + '/t.npy'
        cv_path = data_dir + '/cv.npy'
        G_path = data_dir + '/G.npy'

        # save scalar parameters to json
        with open(params_path, 'w') as f:
            params = {
                'num_nodes': self.num_nodes,
                'num_edges': self.num_edges,
                'num_required_edges': self.num_required_edges,
                'num_vehicles': self.num_vehicles,
                'num_periods': self.num_periods,
                'W': self.W,
                'T_max': self.T_max,
                'M': self.M,
                'theta': self.theta,
                'sigma': self.sigma,
                'ul': self.ul,
                'uu': self.uu
            }
            json.dump(params, f)

        # save array parameters to binary files
        np.save(c_path, self.c)
        np.save(d_path, self.d)
        np.save(t_path, self.t)
        np.save(cv_path, self.cv)
        np.save(G_path, self.G)


class MosaMoiwoaSolverParams:
    """
    Class to store parameters of the solver.
    """

    def __init__(self):
        # initialize default values
        self.N_0: int = 10  # initial number of solutions
        self.MOSA_T_0: float = 800.0  # initial temperature
        self.MOSA_max_iter: int = 200  # maximum number of iterations
        self.MOSA_max_non_improving_iter: int = 10  # maximum number of non-improving iterations
        self.MOSA_alpha: float = 0.9  # cooling factor
        self.MOSA_K: float = 70.0  # Boltzman constant for acceptance probability
        self.MOIWOA_S_min: float = 9.0  # minimum number of children seeds
        self.MOIWOA_S_max: float = 200.0  # maximum number of children seeds
        self.MOIWOA_N_max: int = 100  # maximum number of solutions
        self.MOIWOA_max_iter: int = 300  # maximum number of iterations

    def load_from_dir(self, params_path):
        # load parameters from json
        with open(params_path) as f:
            params = json.load(f)
            if params['N_0'] is not None:
                self.N_0 = params['N_0']
            if params['MOSA_T_0'] is not None:
                self.MOSA_T_0 = params['MOSA_T_0']
            if params['MOSA_max_iter'] is not None:
                self.MOSA_max_iter = params['MOSA_max_iter']
            if params['MOSA_max_non_improving_iter'] is not None:
                self.MOSA_max_non_improving_iter = params['MOSA_max_non_improving_iter']
            if params['MOSA_alpha'] is not None:
                self.MOSA_alpha = params['MOSA_alpha']
            if params['MOSA_K'] is not None:
                self.MOSA_K = params['MOSA_K']
            if params['MOIWOA_S_min'] is not None:
                self.MOIWOA_S_min = params['MOIWOA_S_min']
            if params['MOIWOA_S_max'] is not None:
                self.MOIWOA_S_max = params['MOIWOA_S_max']
            if params['MOIWOA_N_max'] is not None:
                self.MOIWOA_N_max = params['MOIWOA_N_max']
            if params['MOIWOA_max_iter'] is not None:
                self.MOIWOA_max_iter = params['MOIWOA_max_iter']

        # check coherence of parameters
        check_MosaMoiwoaSolverParams(self)

    def save_to_dir(self, params_path):
        # save parameters to json
        with open(params_path, 'w') as f:
            params = {
                'N_0': self.N_0,
                'MOSA_T_0': self.MOSA_T_0,
                'MOSA_max_iter': self.MOSA_max_iter,
                'MOSA_max_non_improving_iter': self.MOSA_max_non_improving_iter,
                'MOSA_alpha': self.MOSA_alpha,
                'MOSA_K': self.MOSA_K,
                'MOIWOA_S_min': self.MOIWOA_S_min,
                'MOIWOA_S_max': self.MOIWOA_S_max,
                'MOIWOA_N_max': self.MOIWOA_N_max,
                'MOIWOA_max_iter': self.MOIWOA_max_iter
            }
            json.dump(params, f)


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
        # compute number of required edges for this period
        num_required_edges = np.sum(problem_params.d > 0)

        # initialize first and second part
        self.first_part = np.full(num_required_edges, -1)
        self.second_part = np.full(num_required_edges, -1)

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
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance

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
