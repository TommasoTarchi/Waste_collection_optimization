import json
import numpy as np
from typing import Tuple, Dict
import networkx as nx
import random


def compute_good_parameters(num_edges: int,
                            num_required_edges: int,
                            num_vehicles: int,
                            max_d: float,
                            max_t: float,
                            avg_cv: float,
                            avg_c: float) -> Dict:
    """
    Function to compute good parameters for the dataset generation, given the problem size,
    in a deterministic way (ideal for scalability study).

    Computed parameters:
    - T_max: computed in such a way that total work time of a vehicle is 'sufficiently'
      smaller
    - ul, uu: computed in such a way that bound (12) of paper can be easily satisfied
    - W: computed in such a way that bound (7) of paper can be easily satisfied
    - theta: computed in such a way that the two terms of the main objective function are
      about the same order of magnitude
    """
    good_params = {}

    # estimate P as 0.4 of number of required edges
    P = 0.1 * num_required_edges

    # compute good T_max
    reference_T_max = 1.2 * P * num_edges * max_t / num_vehicles
    good_params["T_max"] = 0.7 * reference_T_max

    # compute good ul and uu
    max_ul_plus_uu = (reference_T_max * num_vehicles - P * num_edges * max_t) / (num_required_edges * max_d)
    ul_plus_uu = 0.8 * max_ul_plus_uu
    good_params["ul"] = 0.6 * ul_plus_uu
    good_params["uu"] = 0.4 * ul_plus_uu

    # compute good W
    max_W = num_required_edges * max_d / num_vehicles
    good_params["W"] = 0.5 * max_W

    # compute good theta
    good_params["theta"] = avg_cv / (num_edges * avg_c)

    return good_params


def compute_good_parameters_random(num_edges: int,
                                   num_required_edges: int,
                                   num_vehicles: int,
                                   max_d: float,
                                   max_t: float,
                                   avg_cv: float,
                                   avg_c: float) -> Dict:
    """
    Function to compute good parameters for the dataset generation, given the problem size,
    with randomness.

    Computed parameters:
    - T_max: computed in such a way that total work time of a vehicle is 'sufficiently'
      smaller
    - ul, uu: computed in such a way that bound (12) of paper can be easily satisfied
    - W: computed in such a way that bound (7) of paper can be easily satisfied
    - theta: computed in such a way that the two terms of the main objective function are
      about the same order of magnitude
    """
    good_params = {}

    # estimate P as 0.4 of number of required edges
    P = 0.1 * num_required_edges

    # compute good T_max
    max_T_max = 1.2 * P * num_edges * max_t / num_vehicles
    bound_T_max_inf = 0.9 * max_T_max
    bound_T_max_sup = max_T_max
    reference_T_max = random.uniform(bound_T_max_inf, bound_T_max_sup)
    good_T_max = random.uniform(0.8 * reference_T_max, 1.2 * reference_T_max)
    good_params["T_max"] = good_T_max

    # compute good ul and uu
    max_ul_plus_uu = (reference_T_max * num_vehicles - P * num_edges * max_t) / (num_required_edges * max_d)
    bound_ul_plus_uu_inf = 0.7 * max_ul_plus_uu
    bound_ul_plus_uu_sup = 0.9 * max_ul_plus_uu
    ul_plus_uu = random.uniform(bound_ul_plus_uu_inf, bound_ul_plus_uu_sup)
    good_ul = random.uniform(0.4 * ul_plus_uu, 0.6 * ul_plus_uu)
    good_uu = ul_plus_uu - good_ul
    good_params["ul"] = good_ul
    good_params["uu"] = good_uu

    # compute good W
    max_W = num_required_edges * max_d / num_vehicles
    bound_W_inf = 0.5 * max_W
    bound_W_sup = 0.7 * max_W
    good_params["W"] = random.uniform(bound_W_inf, bound_W_sup)

    # compute good theta
    avg_theta = avg_cv / (num_edges * avg_c)
    bound_theta_inf = 0.5 * avg_theta
    bound_theta_sup = 2 * avg_theta
    good_params["theta"] = random.uniform(bound_theta_inf, bound_theta_sup)

    return good_params


def generate_c(num_vertices: int,
               num_edges: int,
               low: float = 1.0,
               high: float = 4.0) -> Tuple[np.ndarray, list]:
    # check inputs
    assert num_vertices > 0, "Number of verticeses must be positive."
    assert low < high, "Low value must be lower than high value."
    assert low > 0, "Low value must be positive."
    if num_edges < num_vertices - 1 or num_edges > num_vertices * (num_vertices - 1) // 2:
        raise ValueError(f"Invalid number of edges. Must be between {num_vertices - 1} and {num_vertices * (num_vertices - 1) // 2}.")

    G = nx.Graph()

    # add a path between the first and last vertices
    available_nodes = set(range(1, num_vertices - 1))
    G.add_edge(0, random.choice(list(available_nodes)))

    for node in range(1, num_vertices - 1):
        if node not in G.nodes:
            G.add_edge(node, random.choice(list(G.nodes)))

    # ensure the last vertex is connected
    G.add_edge(num_vertices - 1, random.choice(list(G.nodes)))

    # add random edges until num_edges is reached
    while len(G.edges()) < num_edges:
        u, v = random.sample(range(num_vertices), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    # assign random distances to edges within the specified bounds
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.uniform(low, high)

    # convert the graph to an adjacency matrix as a NumPy array
    adj_matrix = nx.to_numpy_array(G, weight='weight')

    # get the list of existing edges
    edges_list = []
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if adj_matrix[i, j] > 0:
                edges_list.append((i, j))

    return adj_matrix, edges_list


def generate_d(num_vertices: int,
               existing_edges: list,
               num_required_edges: int,
               low: float = 1.0,
               high: float = 4.0) -> np.ndarray:
    # check inputs
    assert num_vertices > 0, "Number of verticeses must be positive."
    assert num_required_edges > 0, "Number of required edges must be positive."
    assert num_required_edges <= len(existing_edges), "Number of required edges too high w.r.t. number of existing edges."
    assert len(existing_edges) <= num_vertices * (num_vertices - 1) / 2, "Number of existing edges too high w.r.t. number of vertices."
    assert low < high, "Low value must be lower than high value."
    assert low > 0, "Low value must be positive."

    d = np.zeros((num_vertices, num_vertices))

    # randomly choose required edges
    chosen_indices = np.random.choice(len(existing_edges), num_required_edges, replace=False)
    selected_indices = [existing_edges[i] for i in chosen_indices]

    # extract demand values
    row_indices, col_indices = zip(*selected_indices)
    random_values = np.random.uniform(low, high, num_required_edges)
    d[row_indices, col_indices] = random_values

    return d + d.T


def generate_t_G(num_vertices: int,
                 existing_edges: list,
                 low: float = 1.0,
                 high: float = 4.0) -> np.ndarray:
    # check inputs
    assert num_vertices > 0, "Number of verticeses must be positive."
    assert len(existing_edges) <= num_vertices * (num_vertices - 1) / 2, "Number of existing edges too high w.r.t. number of vertices."
    assert low < high, "Low value must be lower than high value."
    assert low > 0, "Low value must be positive."

    t_G = np.zeros((num_vertices, num_vertices))

    # extract demand values
    row_indices, col_indices = zip(*existing_edges)
    random_values = np.random.uniform(low, high, len(existing_edges))
    t_G[row_indices, col_indices] = random_values

    return t_G + t_G.T


def generate_cv(num_vehicles: int, low: float = 1.0, high: float = 4.0) -> np.ndarray:
    # check inputs
    assert num_vehicles > 0, "Number of vehicles must be positive."
    assert low < high, "Low value must be lower than high value."
    assert low > 0, "Low value must be positive."

    return np.random.uniform(low, high, num_vehicles)


def generate_dataset(data_dir: str,
                     bounds_c: tuple,
                     bounds_d: tuple,
                     bounds_t: tuple,
                     bounds_cv: tuple,
                     bounds_G: tuple) -> None:
    """
    Function to generate dataset starting from problem size (read from JSON).

    Parameters:
    bounds are used to compute edges distance, demand, traversing time, usage cost and
    pollution emitted.
    """
    # read problem size from JSON
    num_nodes = None
    num_edges = None
    num_required_edges = None
    num_vehicles = None
    num_periods = None
    with open(data_dir + '/problem_parameters.json') as f:
        params = json.load(f)
        num_nodes = params['num_nodes']
        num_edges = params['num_edges']
        num_required_edges = params['num_required_edges']
        num_vehicles = params['num_vehicles']
        num_periods = params['num_periods']

    # generate dataset
    c, existing_edges = generate_c(num_nodes, num_edges, bounds_c[0], bounds_c[1])

    d = np.zeros((num_nodes, num_nodes, num_periods))
    for t_ in range(num_periods):
        d[:, :, t_] = generate_d(num_nodes, existing_edges, num_required_edges, bounds_d[0], bounds_d[1])

    t = generate_t_G(num_nodes, existing_edges, bounds_t[0], bounds_t[1])
    cv = generate_cv(num_vehicles, bounds_cv[0], bounds_cv[1])
    G = generate_t_G(num_nodes, existing_edges, bounds_G[0], bounds_G[1])

    # save dataset
    c_path = data_dir + '/c.npy'
    d_path = data_dir + '/d.npy'
    t_path = data_dir + '/t.npy'
    cv_path = data_dir + '/cv.npy'
    G_path = data_dir + '/G.npy'

    np.save(c_path, c)
    np.save(d_path, d)
    np.save(t_path, t)
    np.save(cv_path, cv)
    np.save(G_path, G)
