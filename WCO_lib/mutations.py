import numpy as np
import copy


def detect_trips(solution) -> np.ndarray:
    """
    Detect trips in a solution (actually not real trips but sequences of
    edges served by the same vehicle).

    Returns an array in which the trips are identified by the same integer.
    """
    trips = np.zeros(solution.first_part.shape[0], dtype=int)  # initialize trips identifier

    # detect trips
    trips[0] = 0
    for i in range(1, solution.first_part.shape[0]):
        if solution.second_part[i] == solution.second_part[i-1]:
            trips[i] = trips[i-1]
        else:
            trips[i] = trips[i-1] + 1

    return trips


def edge_swap(solution) -> None:
    """
    Swap two edges in a solution randomly.
    """
    first_part = copy.deepcopy(solution.first_part)

    # choose two edges randomly
    i, j = np.random.choice(first_part.shape[0], size=2, replace=False)

    # swap edges and update solution
    first_part[i], first_part[j] = first_part[j], first_part[i]
    solution.set_first_part(first_part)


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
    shuffled = copy.deepcopy(solution.first_part)
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
    flipped = copy.deepcopy(solution.first_part)
    flipped[indexes_trip[0]:indexes_trip[-1] + 1] = np.flip(flipped[indexes_trip[0]:
                                                                    indexes_trip[-1] + 1])

    # update solution
    solution.set_first_part(flipped)


def trip_combine(solution) -> None:
    """
    Combine two halves of two trips in a solution randomly.
    """
    # detect distinct trips
    trips = detect_trips(solution)

    # choose two trips randomly and find positions in the solution array
    trip1, trip2 = np.random.choice(np.unique(trips), size=2, replace=False)

    indexes_trip1 = np.where(trips == trip1)[0]
    indexes_trip2 = np.where(trips == trip2)[0]

    start_trip1 = indexes_trip1[0]
    length_trip1 = indexes_trip1.shape[0]
    lfh_trip1 = int(length_trip1 / 2)  # length of first half of trip1
    start_trip2 = indexes_trip2[0]
    length_trip2 = indexes_trip2.shape[0]
    lfh_trip2 = int(length_trip2 / 2)  # length of first half of trip2

    # make sure trip2 comes after trip1 (needed for correct swapping)
    if start_trip1 > start_trip2:
        start_trip1, start_trip2 = start_trip2, start_trip1
        length_trip1, length_trip2 = length_trip2, length_trip1
        lfh_trip1, lfh_trip2 = lfh_trip2, lfh_trip1

    # initialize new arrays
    old_first_part = copy.deepcopy(solution.first_part)
    combined_first_part = np.empty_like(old_first_part)

    old_second_part = copy.deepcopy(solution.second_part)
    combined_second_part = np.empty_like(old_second_part)

    # create new array with combined trips
    combined_first_part[:start_trip1] = old_first_part[:start_trip1]
    combined_first_part[start_trip1:start_trip1 + lfh_trip2] = old_first_part[start_trip2:start_trip2 + lfh_trip2]
    combined_first_part[start_trip1 + lfh_trip2:start_trip2 + lfh_trip2 - lfh_trip1] = old_first_part[start_trip1 + lfh_trip1:
                                                                                                      start_trip2]
    combined_first_part[start_trip2 + lfh_trip2 - lfh_trip1:start_trip2 + lfh_trip2] = old_first_part[start_trip1:
                                                                                                      start_trip1 + lfh_trip1]
    combined_first_part[start_trip2 + lfh_trip2:] = old_first_part[start_trip2 + lfh_trip2:]

    # get vehicles of the two trips
    vehicle1 = old_second_part[start_trip1]
    vehicle2 = old_second_part[start_trip2]

    # update vehicles in second part
    combined_second_part[:start_trip1] = old_second_part[:start_trip1]
    combined_second_part[start_trip1:start_trip1 + lfh_trip2] = vehicle1
    combined_second_part[start_trip1 + lfh_trip2:start_trip2 + lfh_trip2 - lfh_trip1] = old_second_part[start_trip1 + lfh_trip1:
                                                                                                        start_trip2]
    combined_second_part[start_trip2 + lfh_trip2 - lfh_trip1:start_trip2 + lfh_trip2] = vehicle2
    combined_second_part[start_trip2 + lfh_trip2:] = old_second_part[start_trip2 + lfh_trip2:]

    # update solution
    solution.set_first_part(combined_first_part)
    solution.set_second_part(combined_second_part)
