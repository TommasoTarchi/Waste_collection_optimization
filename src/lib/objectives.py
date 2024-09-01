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
