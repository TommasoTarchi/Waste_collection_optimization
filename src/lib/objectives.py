def update_service_time(old_time, edge_demand, edge_traversing_time, ul, uu):
    """
    Update the service time of a vehicle based on last edge traversed.
    """

    new_time = old_time + edge_demand * (ul + uu) + edge_traversing_time

    return new_time


def update_capacity(current_capacity, edge_demand):
    """
    Update the capacity of a vehicle based on last edge traversed.
    """

    new_capacity = current_capacity - edge_demand

    return new_capacity
