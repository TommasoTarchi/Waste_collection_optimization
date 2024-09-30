from typing import Tuple, List
import itertools
import gurobipy as gb


def find_subsets_old(edge_list) -> Tuple:
    """
    Function to find all subsets of the given set of edges.
    Also returns the list of indices of vertices in and not in each subset.

    NOTICE: (i, j) and (j, i) are always placed in the same subset.
    """
    subsets = []  # list of subsets
    vertices = []  # list of vertices in subsets
    vertices_not = []  # list of vertices not in subsets

    # find unqiue edges
    edge_list_unique = []
    for (i, j) in edge_list:
        if (j, i) in edge_list_unique:
            continue
        edge_list_unique.append((i, j))

    # find all vertices
    all_vertices = [v for edge in edge_list_unique for v in edge]

    # find all subsets with corresponding indices
    for num_edges in range(1, len(edge_list_unique) + 1):
        sub = itertools.combinations(edge_list_unique, num_edges)
        for edge_set in sub:
            subsets.append(edge_set)
            vertices.append([])
            for edge in list(edge_set):
                if not edge[0] in vertices[-1]:
                    vertices[-1].append(edge[0])
                if not edge[1] in vertices[-1]:
                    vertices[-1].append(edge[1])

    # find indices not in subsets
    for vertices_set in vertices:
        vertices_not.append([vertex for vertex in all_vertices if vertex not in vertices_set])

    # add 'inverse' edges to subsesubsets
    for subset in subsets:
        for (i, j) in subset:
            subset.append((j, i))

    return (subsets, vertices, vertices_not)


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

    print(subsets)

    return subsets, vertices_in, vertices_not


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
                subsets, subsets_vertices, subsets_vertices_not = find_subsets(relevant_edges)

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


# TODO: rimuovere violation e altra roba per debugging
def add_subtours_constraint_all(model: gb.Model,
                                x: gb.tupledict,
                                num_nodes: int,
                                existing_edges: List,
                                K: int,
                                P: int,
                                T: int,
                                M: int):
    """
    Function to add all subtours constraints to model.
    """
    #violation = model.addVar(name="violation_13")
    violation = 0

    for t in range(T):
        for p in range(P):
            for k in range(K):
                # find dubsets with corresponding vertices
                subsets, subsets_vertices, subsets_vertices_not = find_subsets(existing_edges)

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
                    #model.addConstr(left_hand_side <= M * right_hand_side + 1e-6)
                    #model.addConstr(left_hand_side <= M * right_hand_side + violation)

                    #model.optimize()
                    #if model.Status == gb.GRB.INFEASIBLE:
                    #    print("Model became infeasible after adding constraints for subset: ", subsets[idx])
                    #    break

    return violation
