import numpy as np


def detect_trips(solution) -> np.ndarray:
    """
    Detect trips in a solution (actually not real trips but sequences of
    edges served by the same vehicle).

    Returns an array in which the trips are identified by the same integer.
    """
    trips = np.zeros(solution.first_part.size[0], dtype=int)  # initialize trips identifier

    # detect trips
    trips[0] = 0
    for i in range(1, solution.first_part.size[0]):
        if solution.second_part[i] == solution.second_part[i-1]:
            trips[i] = trips[i-1]
        else:
            trips[i] = trips[i-1] + 1

    return trips


def edge_swap(solution) -> None:
    """
    Swap two edges in a solution randomly.
    """
    first_part = solution.first_part.copy()

    # choose two edges randomly
    i, j = np.random.choice(first_part.size[0], size=2, replace=False)

    # swap edges and update solution
    first_part[i], first_part[j] = first_part[j], first_part[i]
    solution.set_first_part(first_part)


def trip_swap(solution) -> None:
    """
    Swap two trips in a solution randomly.
    """
    # detect distinct trips
    trips = detect_trips(solution)

    # choose two trips randomly and find positions in the solution array
    trip1, trip2 = np.random.choice(np.unique(trips), size=2, replace=False)

    indexes_trip1 = np.where(trips == trip1)[0]
    indexes_trip2 = np.where(trips == trip2)[0]

    start_trip1 = indexes_trip1[0]
    length_trip1 = indexes_trip1.shape[0]
    start_trip2 = indexes_trip2[0]
    length_trip2 = indexes_trip2.shape[0]

    # initialize new array
    old_array = solution.first_part.copy()
    swapped = np.empty_like(old_array)

    # create new array with swapped trips
    swapped[:start_trip1] = old_array[:start_trip1]
    swapped[start_trip1:start_trip1 + length_trip2] = old_array[start_trip2:
                                                                start_trip2 + length_trip2]
    swapped[start_trip1 + length_trip2:start_trip2 - length_trip1 + length_trip2] = old_array[start_trip1 + length_trip1:
                                                                                              start_trip2]
    swapped[start_trip2 - length_trip1 + length_trip2:start_trip2 + length_trip2] = old_array[start_trip1:
                                                                                              start_trip1 + length_trip1]
    swapped[start_trip2 + length_trip2:] = old_array[start_trip2 + length_trip2:]

    # update solution
    solution.set_first_part(swapped)


def trip_shuffle(solution) -> None:
    """
    Shuffle order of served edges in a trip in a solution randomly.
    """
    # detect distinct trips
    trips = detect_trips(solution)

    # choose a trip randomly and find positions in the solution array
    trip = np.random.choice(np.unique(trips), size=1, replace=False)
    indexes_trip = np.where(trips == trip)[0]

    # shuffle the sequence of edges in the trip
    shuffled = solution.first_part.copy()
    np.random.shuffle(shuffled[indexes_trip[0]:indexes_trip[-1] + 1])

    # update solution
    solution.set_first_part(shuffled)


def trip_reverse(solution) -> None:
    """
    Reverse order of served edges in a trip in a solution randomly.
    """
    # detect distinct trips
    trips = detect_trips(solution)

    # choose a trip randomly and find positions in the solution array
    trip = np.random.choice(np.unique(trips), size=1, replace=False)
    indexes_trip = np.where(trips == trip)[0]

    # flip the sequence of edges in the trip
    flipped = solution.first_part.copy()
    flipped[indexes_trip[0]:indexes_trip[-1] + 1] = np.flip(flipped[indexes_trip[0]:
                                                                    indexes_trip[-1] + 1])

    # update solution
    solution.set_first_part(flipped)


def trip_combine(solution) -> None:
    """
    Combine two halves of two trips in a solution randomly.
    """
    # TODO
    pass
