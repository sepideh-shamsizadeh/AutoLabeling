import numpy as np
from scipy.spatial.distance import cdist, mahalanobis


def global_nearest_neighbor(reference_points, query_points, covariance_matrix):
    reference_points = np.array(reference_points)
    distances = cdist(reference_points, query_points, lambda u, v: mahalanobis(u, v, covariance_matrix))
    neighbors = []
    for i, dis in enumerate(distances):
        if dis <= 0.7:
            neighbors.append(i)
    return neighbors


def state_transition_fn(x, dt):
    # Implement the state transition function
    # x: current state vector [x, y, vx, vy]
    # dt: time step
    # Return the predicted state vector
    # Example: simple constant velocity model
    return np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]])


# Define the measurement function
def measurement_fn(x):
    # Implement the measurement function
    # x: current state vector [x, y, vx, vy]
    # Return the measurement vector
    # Example: position measurement
    return x[:2]


def handle_loss_of_id(filters, remove_filters):
    # Remove the filter from the list of filters
    for f in remove_filters:
        # print(f.object_id)
        filters.remove(f)
    return filters