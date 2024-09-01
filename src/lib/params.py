import json
import numpy as np

from .objectives import compute_service_time, update_capacity


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
    assert np.sum(params.d > 0) == params.num_required_edges, "Number of d > 0 not coherent with problem size."
    assert len(params.required_edges) == params.num_required_edges, "Number of required edges not coherent with problem size."

    # check demand is non-negative
    assert np.all(params.d >= 0), "Demand d must be non-negative."


class Params:
    """
    Class to store parameters of the problem.
    """

    def __init__(self):
        # size of the problem
        self.num_nodes = 0  # number of nodes
        self.num_edges = 0  # number of edges
        self.num_required_edges = 0  # number of required edges (if (i,j) and (j,i) are
                                     # required, only one is counted)
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

    def load_from_dir(self, data_path):
        # define paths to init data
        params_path = data_path + 'scalar.json'
        c_path = data_path + 'c.npy'
        d_path = data_path + 'd.npy'
        t_path = data_path + 't.npy'
        cv_path = data_path + 'cv.npy'
        G_path = data_path + 'G.npy'

        # load scalar parameters from json
        with open(params_path) as f:
            params = json.load(f)
            self.num_nodes = params['num_nodes']
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

        # compute total number of edges
        self.num_edges = self.num_nodes * self.num_nodes

        # compute required edges coordinates
        required_coord = []
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if self.d[i, j] > 0:
                    required_coord.append((i, j))
        self.required_edges = required_coord

        # check coherence of parameters
        check_params(self)

    def save_to_dir(self, data_path):
        # define paths to init data
        params_path = data_path + 'scalar.json'
        c_path = data_path + 'c.npy'
        d_path = data_path + 'd.npy'
        t_path = data_path + 't.npy'
        cv_path = data_path + 'cv.npy'
        G_path = data_path + 'G.npy'

        # save scalar parameters to json
        with open(params_path, 'w') as f:
            params = {
                'num_nodes': self.num_nodes,
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


class Solution:
    """
    Class to store solution of the problem in vector format.
    """

    def __init__(self):
        self.first_part = None
        self.second_part = None
        self.service_times = None
        self.total_service_time = None
        self.total_travelled_distance = None

    def set_first_part(self, first_part: np.ndarray):
        assert len(first_part.shape) == 1, "First part must be a 1D vector."

        # set first part of the solution to a given vector
        self.first_part = first_part

    def set_second_part(self, second_part: np.ndarray):
        assert len(second_part.shape) == 1, "Second part must be a 1D vector."

        # set second part of the solution to a given vector
        self.second_part = second_part

    def shuffle_first_part(self):
        # shuffle first part of the solution
        np.random.shuffle(self.first_part)

    def compute_second_part(self, params: Params):
        # check coherence
        assert self.first_part is not None, "First part must be set before computing second part."
        assert self.first_part.shape[0] == params.num_required_edges, "First part must have the same size as the number of required nodes - 2."

        # compute second part
        # TODO

    def init_heuristic(self, params: Params):

        # init first and second part
        self.first_part = np.full(params.num_required_edges, -1)
        self.second_part = np.full(params.num_required_edges, -1)

        # init service times and capacities
        service_times = np.zeros(params.num_vehicles)
        capacities = np.full(params.num_vehicles, params.W)

        # init travelled distance
        travelled_distance = 0

        # init vehicle positions
        positions = np.full(params.num_vehicles, 0)

        # init available vehicles
        available_vehicles = np.full(params.num_vehicles, True)

        # select random vehicle and mark as not available
        current_vehicle = np.random.randint(params.num_vehicles)
        available_vehicles[current_vehicle] = False

        # iterate until all required edges are covered
        d_temp = params.d.copy()
        required_edges_temp = params.required_edges.copy()
        solution_idx = 0  # index of the current element of the solution
        while required_edges_temp:
            # select current position
            current_position = positions[current_vehicle]

            # find closest starting node of a required edge to current position
            candidate_next_starts = np.where(np.any(d_temp > 0, axis=1))[0]
            next_start = candidate_next_starts[np.argmin(params.c[current_position, candidate_next_starts])]

            # choose ending node of required edge at random
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
                # update service time and capacity
                service_times[current_vehicle] = next_service_time
                capacities[current_vehicle] = next_capacity

                # update demand
                d_temp[next_start, next_end] = 0

                # update position
                positions[current_vehicle] = next_end

                # update travelled distance
                travelled_distance += params.c[current_position, next_start] + params.c[next_start, next_end]

                # update required edges (symmetrically)
                if next_end < next_start:
                    next_start, next_end = next_end, next_start
                required_edges_temp.remove((next_start, next_end))

                # update elements of the solution
                self.first_part[solution_idx] = params.required_edges.index((next_start, next_end))
                self.second_part[solution_idx] = current_vehicle
                solution_idx += 1

            # go to disposal site
            elif current_position != params.num_nodes-1:
                # update service time and capacity
                service_times[current_vehicle] = params.t[current_position, params.num_nodes-1]
                capacities[current_vehicle] = params.W

                # update position
                positions[current_vehicle] = params.num_nodes-1

                # update travelled distance
                travelled_distance += params.t[current_position, params.num_nodes-1]

            # go back to depot
            else:
                # update service time
                service_times[current_vehicle] = params.t[params.num_nodes-1, 0]

                # update position
                positions[current_vehicle] = 0

                # update travelled distance
                travelled_distance += params.t[params.num_nodes-1, 0]

                # select new vehicle among the available ones
                current_vehicle = np.random.choice(np.where(available_vehicles)[0])
                available_vehicles[current_vehicle] = False

        # move all vehicles to depot and update service times
        for vehicle in range(params.num_vehicles):
            if positions[vehicle] != 0:
                service_times[vehicle] += params.t[positions[vehicle], params.num_nodes-1] + params.t[params.num_nodes-1, 0]

        # save service times
        self.service_times = service_times
        self.total_service_time = np.sum(service_times)

        # save total travelled distance
        self.total_travelled_distance = travelled_distance
