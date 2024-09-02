import json
import numpy as np


def check_params(params):
    """
    Check if the parameters are coherent.
    """
    # check the size of arrays
    assert params.c.shape == (params.num_nodes, params.num_nodes), "Shape of c not coherent with problem size."
    assert params.d.shape == (params.num_nodes, params.num_nodes, params.num_periods), "Shape of d not coherent with problem size."
    assert params.t.shape == (params.num_nodes, params.num_nodes), "Shape of t not coherent with problem size."
    assert params.cv.shape == (params.num_vehicles), "Shape of cv not coherent with problem size."
    assert params.G.shape == (params.num_nodes, params.num_nodes), "Shape of G not coherent with problem size."

    # check demand is non-negative
    assert np.all(params.d >= 0), "Demand d must be non-negative."


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
    Class to store parameters of the problem (except for demand).
    """

    def __init__(self):
        # size of the problem
        self.num_nodes = 0  # number of nodes
        self.num_edges = 0  # number of edges
        self.num_vehicles = 0  # number of vehicles
        self.num_periods = 0  # number of planning periods

        # parameters of the problem
        self.c = None  # edge distance (not necessarily symmetric)
        self.W = 0  # vehicle capacity
        self.d = None  # edge demand at each period
        self.T_max = 0  # maximum available time for vehicles
        self.M = 0  # a large number
        self.t = None  # traversing time of edges
        self.cv = None  # usage cost of vehicles
        self.theta = 0  # conversion factor of distance to cost
        self.G = None  # pollution emitted by traversing edges
        self.sigma = 0  # number of workforce per vehicle
        self.ul = 0  # conversion factor of demand to loading time
        self.uu = 0  # conversion factor of demand to unloading time

        # required edges coordinates (i is staring point and j is ending point)
        self.required_edges = None

        # number of periods each vehicle is employed for
        self.periods_employed = None

    def load_from_dir(self, data_path):
        # define paths to init data
        params_path = data_path + '/scalar.json'
        c_path = data_path + '/c.npy'
        d_path = data_path + '/d.npy'
        t_path = data_path + '/t.npy'
        cv_path = data_path + '/cv.npy'
        G_path = data_path + '/G.npy'

        # load scalar parameters from json
        with open(params_path) as f:
            params = json.load(f)
            self.num_nodes = params['num_nodes']
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

        # compute total number of edges
        self.num_edges = self.num_nodes * self.num_nodes

        # compute required edges coordinates for each period (list of
        # lists of tuples)
        required_coord = []
        for _ in range(self.num_periods):
            required_coord.append([])
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if self.d[i, j] > 0:
                        required_coord[-1].append((i, j))
        self.required_edges = required_coord

        # init periods periods employed
        self.periods_employed = np.zeros(self.num_vehicles)

        # check coherence of parameters
        check_params(self)

    def save_to_dir(self, data_path):
        # define paths to init data
        params_path = data_path + '/scalar.json'
        c_path = data_path + '/c.npy'
        d_path = data_path + '/d.npy'
        t_path = data_path + '/t.npy'
        cv_path = data_path + '/cv.npy'
        G_path = data_path + '/G.npy'

        # save scalar parameters to json
        with open(params_path, 'w') as f:
            params = {
                'num_nodes': self.num_nodes,
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


class SinglePeriodSolution:
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

    def adjust_first_part(self, params: ProblemParams):
        """
        Adjust the first part of the solution to satisfy constraints.
        """
        # check coherence
        assert self.second_part is not None, "Second part must be set before the first part is adjusted."

        # compute first part
        # TODO

    def adjust_second_part(self, params: ProblemParams):
        """
        Adjust the second part of the solution to satisfy constraints.
        """
        # check coherence
        assert self.first_part is not None, "First part must be set before the second part is adjusted."

        # compute second part
        # TODO

    def init_heuristic(self, params: ProblemParams):
        """
        Initialize the solution with a heuristic.
        """
        # compute number of required edges for this period
        num_required_edges = np.sum(params.d > 0)

        # initialize first and second part
        self.first_part = np.full(num_required_edges, -1)
        self.second_part = np.full(num_required_edges, -1)

        # initialize auxiliary variables
        service_times = np.zeros(params.num_vehicles)
        capacities = np.full(params.num_vehicles, params.W)
        traversals = np.zeros_like(params.c)  # number of traversals of each edge
        travelled_distance = 0  # total travelled distance
        positions = np.full(params.num_vehicles, 0)  # current position of each vehicle
        available_vehicles = np.full(params.num_vehicles, True)

        # select random vehicle and mark as not available
        current_vehicle = np.random.randint(params.num_vehicles)
        available_vehicles[current_vehicle] = False

        # initialize vehicle employment
        self.vehicle_employed = np.full(params.num_vehicles, False)

        # iterate until all required edges are covered
        d_temp = params.d[:, :, self.period].copy()
        required_edges_temp = params.required_edges.copy()
        solution_idx = 0  # index of the current element of the solution
        while required_edges_temp:
            # select current position
            current_position = positions[current_vehicle]

            # find closest starting node of a required edge to current position
            candidate_next_starts = np.where(np.any(d_temp > 0, axis=1))[0]
            next_start = candidate_next_starts[np.argmin(params.c[current_position, candidate_next_starts])]

            # choose ending node of required edge at random among non-zero demand edges
            next_end = np.random.choice(np.nonzero(next_start)[0])

            # compute service time for possible next position
            next_service_time = params.c[current_position, next_start]  # time to go to required edge
            next_service_time += compute_service_time(params.d[next_start, next_end],
                                                      params.t[next_start, next_end],
                                                      params.ul,
                                                      params.uu)  # add time for service of required edge
            next_service_time_tot = params.t[next_end, params.num_nodes-1]  # add time to go to disposal site
            next_service_time_tot += params.t[params.num_nodes-1, 0]  # add time to go back to depot

            # compute remaining capacity for possible next position
            next_capacity = update_capacity(capacities[current_vehicle], params.d[next_start, next_end])

            # serve next required edge with current vehicle
            if next_service_time_tot < params.T_max and next_capacity >= 0:
                # update
                service_times[current_vehicle] = next_service_time
                capacities[current_vehicle] = next_capacity
                d_temp[next_start, next_end] = 0
                positions[current_vehicle] = next_end
                traversals[next_start, next_end] += 1
                travelled_distance += params.c[current_position, next_start] + params.c[next_start, next_end]

                # update required edges (symmetrically)
                if next_end < next_start:
                    next_start, next_end = next_end, next_start
                required_edges_temp.remove((next_start, next_end))

                # update elements of the solution
                self.first_part[solution_idx] = params.required_edges.index((next_start, next_end))
                self.second_part[solution_idx] = current_vehicle
                solution_idx += 1

                # mark vehicle as employed
                self.vehicle_employed[current_vehicle] = True

            # go to disposal site
            elif current_position != params.num_nodes-1:
                # update
                service_times[current_vehicle] = params.t[current_position, params.num_nodes-1]
                capacities[current_vehicle] = params.W
                positions[current_vehicle] = params.num_nodes-1
                traversals[current_position, params.num_nodes-1] += 1
                travelled_distance += params.t[current_position, params.num_nodes-1]

            # go back to depot
            else:
                # update
                service_times[current_vehicle] = params.t[params.num_nodes-1, 0]
                positions[current_vehicle] = 0
                traversals[params.num_nodes-1, 0] += 1
                travelled_distance += params.t[params.num_nodes-1, 0]

                # select new vehicle among the available ones
                current_vehicle = np.random.choice(np.where(available_vehicles)[0])
                available_vehicles[current_vehicle] = False

        # move all vehicles to depot and update service times
        for vehicle in range(params.num_vehicles):
            if positions[vehicle] != 0:
                service_times[vehicle] += params.t[positions[vehicle], params.num_nodes-1] + params.t[params.num_nodes-1, 0]

        # save supplementary data
        self.total_service_time = np.sum(service_times)
        self.traversals = traversals
        self.total_travelled_distance = travelled_distance

    def compute_objectives(self, params: ProblemParams):
        """
        Compute the objective functions of the solution.
        """
        # compute total waste collection routing cost
        Z_1 = params.theta * self.total_travelled_distance + params.cv.dot(self.vehicle_employed)

        # compute total pollution routing cost
        Z_2 = np.sum(params.G * self.traversals)

        # compute total amount of hired labor
        Z_3 = params.sigma * np.sum(self.vehicle_employed)

        # compute total work deviation
        Z_4 = np.sum(1 - self.total_service_time / params.T_max)

        return Z_1, Z_2, Z_3, Z_4
