from typing import Tuple, List
import itertools
import gurobipy as gb


def find_all_subsets(edge_list) -> Tuple:
    """
    Function to find all subsets of the given set of edges.
    Also returns the list of indices of vertices in and not in each subset.
    """
    subsets = []  # list of subsets
    vertices = []  # list of vertices in subsets
    vertices_not = []  # list of vertices not in subsets

    # find all subsets with corresponding indices
    for num_edges in range(1, len(edge_list) + 1):
        sub = itertools.combinations(edge_list, num_edges)
        for edge_set in sub:
            subsets.append(edge_set)
            vertices.append([])
            for edge in list(edge_set):
                if not edge[0] in vertices[-1]:
                    vertices[-1].append(edge[0])
                if not edge[1] in vertices[-1]:
                    vertices[-1].append(edge[1])

    # find all vertices
    all_vertices = []
    for edge in edge_list:
        if not edge[0] in all_vertices:
            all_vertices.append(edge[0])
        if not edge[1] in all_vertices:
            all_vertices.append(edge[1])

    # find indices not in subsets
    for vertices_set in vertices:
        vertices_not.append([vertex for vertex in all_vertices if vertex not in vertices_set])

    return (subsets, vertices, vertices_not)


def add_subtours_constraint(model: gb.Model,
                            x: gb.tupledict,
                            num_nodes: int,
                            existing_edges: List,
                            K: int,
                            P: int,
                            T: int,
                            M: int,
                            traversal_fraction_threshold: float = 0.7,
                            traversal_absolut_threshold: int = 6) -> bool:
    """
    Function to conditionally add subtours constraint to model.
    """
    # get best solution so far
    x_values = model.getAttr("x", x)

    added_constr = False  # flag indicating whether constraint was added

    for t in range(T):
        for p in range(P):
            for k in range(K):
                # compute total and maximum number of traversals
                tot_traversal = 0
                max_traversals = 0
                for (i, j) in existing_edges:
                    if x_values[i, j, k, p, t] > max_traversals:
                        max_traversals = x_values[i, j, k, p, t]
                        tot_traversal += x_values[i, j, k, p, t]

                # compute threshold for number of traversals
                traversal_threshold = traversal_fraction_threshold * max_traversals

                # find relevant edges
                relevant_edges = []
                for (i, j) in existing_edges:
                    if x_values[i, j, k, p, t] > traversal_threshold:
                        relevant_edges.append((i, j))

                # skip constraint if all edges have been traversed many times
                if len(relevant_edges) in (0, len(existing_edges)) or (max_traversals / len(relevant_edges)) <= traversal_absolut_threshold:
                    continue

                print("Adding subtour constraint for k = ", k, ", p = ", p, ", t = ", t, " with ", len(relevant_edges), " edges.")

                # find relevant subsets
                subsets, subsets_vertices, subsets_vertices_not = find_all_subsets(relevant_edges)

                # add constraints
                for idx in range(len(subsets)):
                    # compute expressions
                    left_hand_side = gb.LinExpr([(1.0, x[i, j, k, p, t])
                                                 for (i, j) in subsets[idx]])
                    right_hand_side = gb.LinExpr([(1.0, x[i, j, k, p, t])
                                                  for (i, j) in existing_edges
                                                  if i in subsets_vertices_not[idx]
                                                  if j in subsets_vertices[idx]
                                                  if not j == 0 and
                                                  not j == num_nodes-1])

                    # set constraint
                    model.addConstr(left_hand_side <= M * right_hand_side)

                # set flag
                added_constr = True

    return added_constr
