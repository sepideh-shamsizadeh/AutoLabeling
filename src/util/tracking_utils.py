import yaml
import numpy as np
from scipy.spatial.distance import cdist, mahalanobis

def write_output(people, fid, file_name):
    data = []
    for k, p in enumerate(people):
        position = {'x': p[0], 'y': p[1]}
        pp = {'id' + str(k): position}
        data.append(pp)
    yaml_data = {'frame ' + str(fid): data}
    output_file = file_name

    # Open the file in write mode
    with open(output_file, 'a') as file:
        # Write the YAML data to the file
        yaml.dump(yaml_data, file)


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