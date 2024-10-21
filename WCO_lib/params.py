import json
import numpy as np
import networkx as nx


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

        # networkx graph for MOSA-MOIWOA
        self.graph = None

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

    def build_graph(self):
        """
        Build a networkx graph from the problem parameters.
        """
        assert self.existing_edges is not None, "Existing edges must be defined."
        assert self.c is not None, "Edge distance must be defined."

        graph = nx.Graph()

        # add edges to the graph
        graph.add_edges_from(self.existing_edges)
        for i, j in graph.edges():
            graph[i][j]['weight'] = self.c[i, j]

        # save graph
        self.graph = graph

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

    def load_from_file(self, params_path):
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

    def save_to_file(self, params_path):
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
