from typing import Tuple, List
import itertools
import gurobipy as gb
import networkx as nx


def find_subsets(edge_list) -> Tuple:
    """
    Function to find all subsets of the given set of edges.
    Also returns the list of indices of vertices in and not in each subset.

    NOTICE: (i, j) and (j, i) are always placed in the same subset.
    """
    subsets = []
    vertices_in = []
    vertices_not = []

    # create set of undirected edges (both (i,j) and (j,i) for each edge)
    undirected_edges = set()
    for i, j in edge_list:
        undirected_edges.add((min(i, j), max(i, j)))

    # find all vertices
    all_vertices = set(v for edge in undirected_edges for v in edge)

    # find subsets and corresponding vertices
    for r in range(1, len(undirected_edges) + 1):
        for edge_subset in itertools.combinations(undirected_edges, r):
            # Create a symmetric subset
            symmetric_subset = frozenset((i, j) for i, j in edge_subset).union(
                frozenset((j, i) for i, j in edge_subset)
            )

            if symmetric_subset not in subsets:
                subsets.append(symmetric_subset)

                vertices_in_subset = set(v for edge in symmetric_subset for v in edge)
                vertices_in.append(vertices_in_subset)
                vertices_not.append(all_vertices - vertices_in_subset)

    return subsets, vertices_in, vertices_not


def find_connected_subsets(edge_list: list) -> Tuple:
    """
    Function to find all connected subsets of the given set of edges.
    Also returns the list of vertices in and not in each subset.

    NOTICE: (i, j) and (j, i) are always placed in the same subset.
    """
    subsets = []
    vertices_in = []
    vertices_not = []

    # Create set of undirected edges (ensure both (i, j) and (j, i) are treated as the same)
    undirected_edges = set()
    for i, j in edge_list:
        undirected_edges.add((min(i, j), max(i, j)))

    # Find all vertices in the original edge list
    all_vertices = set(v for edge in undirected_edges for v in edge)

    # Function to check if a subset of edges forms a connected graph
    def is_connected(edge_subset):
        G = nx.Graph()
        G.add_edges_from(edge_subset)
        return nx.is_connected(G)

    # Find subsets and corresponding vertices
    for r in range(1, len(undirected_edges) + 1):
        for edge_subset in itertools.combinations(undirected_edges, r):
            symmetric_subset = frozenset((i, j) for i, j in edge_subset).union(
                frozenset((j, i) for i, j in edge_subset)
            )

            if symmetric_subset not in subsets:
                # Check if the symmetric subset is connected
                if is_connected(symmetric_subset):
                    subsets.append(symmetric_subset)

                    vertices_in_subset = set(v for edge in symmetric_subset for v in edge)
                    vertices_in.append(vertices_in_subset)
                    vertices_not.append(all_vertices - vertices_in_subset)

    return subsets, vertices_in, vertices_not


def add_subtours_constraint(model: gb.Model,
                            x: gb.tupledict,
                            num_nodes: int,
                            existing_edges: List,
                            K: int,
                            P: int,
                            T: int,
                            M: int) -> None:
    """
    Function to add subtours constraints to model.
    """
    # find subsets with corresponding vertices
    subsets, subsets_vertices, subsets_vertices_not = find_connected_subsets(existing_edges)

    for t in range(T):
        for p in range(P):
            for k in range(K):
                for idx in range(len(subsets)):
                    # compute expressions
                    left_hand_side = gb.LinExpr([(1.0, x[i, j, k, p, t])
                                                 for (i, j) in subsets[idx]])
                    right_hand_side = gb.LinExpr([(1.0, x[i, j, k, p, t])
                                                  for i in subsets_vertices_not[idx]
                                                  for j in subsets_vertices[idx]
                                                  if (i, j) in existing_edges
                                                  and j not in (0, num_nodes-1)])

                    # add 'incoming' edge when origin of trip is in subset
                    if (p == 0 and 0 in subsets_vertices[idx]) or (p != 0 and num_nodes-1 in subsets_vertices[idx]):
                        right_hand_side += 1

                    # set constraint
                    model.addConstr(left_hand_side <= M * right_hand_side)
