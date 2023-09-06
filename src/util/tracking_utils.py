import numpy as np
from scipy.spatial.distance import cdist, mahalanobis
from src.util.visual_feature_utils import *
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

# Set the parameters
num_states = 4  # Number of states (x, y, vx, vy)
num_measurements = 2  # Number of measurements (position)
process_noise_variance = 0.1  # Process noise variance
measurement_noise_variance = 0.01  # Measurement noise variance
dt = 0.01  # Time step

# Define the process and measurement noise covariance matrices
Q = np.eye(num_states) * process_noise_variance
R = np.eye(num_measurements) * measurement_noise_variance

# Set initial state and covariance matrix
initial_covariance = np.eye(num_states) * 1.0  # Initial covariance matrix

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


def global_nearest_neighbor(reference_points, query_points, covariance_matrix, ids):
    reference_points = np.array(reference_points)
    neighbors = []
    if len(reference_points) > 0 and len(query_points) > 0:
        distances = cdist(reference_points, query_points, lambda u, v: mahalanobis(u, v, covariance_matrix))
        for i, dis in zip(ids, distances):
            if dis <= 1.5:
                neighbors.append(i)
    return neighbors


def add_loss_of_id(filters, missed_id, missed_filters):
    while len(missed_id) > 0:
        mid = missed_id[0]
        missed_id.remove(mid)
        print('filter with id' + str(mid) + 'missed')
        missed_filters[mid] = filters[mid]
        filters.pop(mid)
    return filters, missed_filters


def creat_new_filter(measurement, id):
    person = measurement['position'][0]
    filter_i = UnscentedKalmanFilter(dim_x=num_states, dim_z=num_measurements, dt=dt,
                                     fx=state_transition_fn, hx=measurement_fn,
                                     points=MerweScaledSigmaPoints(num_states, alpha=0.1, beta=2.,
                                                                   kappa=-1.0))

    # Set initial state and covariance matrix
    filter_i.x = [person[0], person[1], 0, 0]
    filter_i.P = initial_covariance
    filter_i.dim_x = num_states

    # Set process and measurement noise covariance matrices
    filter_i.Q = Q
    filter_i.R = R

    # Set object ID
    filter_i.object_id = id
    # Initialize loss of measurement association counter
    filter_i.visual_features = measurement['visual_features'][0]
    filter_i.bounding = measurement['bounding_box'][0]
    return filter_i


def list_generator(input_list):
    for item in input_list:
        yield item


def get_neighbors(positions, filter_i, id_g, measurements):
    covariance_matrix = np.array([[1, 0], [0, 1]])
    neighbors = []
    n = 0
    if len(positions) > 0:
        neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix, id_g)
    if len(neighbors) > 0:
        for i, neighbor in enumerate(neighbors):
            if str(neighbor) in measurements:
                n += 1
            else:
                neighbors.pop(i)
        if n > 0:
            return neighbors
        else:
            return []
    return neighbors


def get_id(ids, assigned, id_g):
    gen = list_generator(ids)
    found_id = next(gen)
    if found_id in assigned:
        try:
            while found_id in assigned:
                found_id = next(gen)
        except Exception as e:
            found_id = -1
    if -1 < found_id < len(id_g):
        found_id = id_g[found_id]
    else:
        found_id = -1
    return found_id


def find_tracker_newF(filter_i, positions, measurements, dt, assigned, id_g):
    ids = []
    neighbors = []
    filter_i.predict(dt=dt)
    found_id = -1
    if len(positions) > 0:
        neighbors = get_neighbors(positions, filter_i, id_g, measurements)
    if len(neighbors) > 0:
        gallery = []
        for neighbor in neighbors:
            gallery.append(measurements[str(neighbor)]['visual_features'][0])
        sim, indices = calculate_similarity_faiss(filter_i.visual_features, gallery)
        for index in indices:
            ids.append(neighbors[index])
        found_id = get_id(ids, assigned, id_g)
        if sim[0] < 0.9:
            found_id = -1
    return found_id


def check_neighbours(filters, measurements, id_rem, assigned, positions, galleries, id_g):
    assigned_filters = []
    missed_id = []
    for filter_id, filter_i in filters.items():
        id_f = find_tracker_newF(filter_i, positions, measurements, dt, assigned, id_g)
        if id_f in id_rem:
            assigned.append(id_f)
            id_rem.remove(id_f)
            filter_i.update(measurements[str(id_f)]['position'][0])
            filter_i.visual_features = measurements[str(id_f)]['visual_features'][0]
            filter_i.bounding = measurements[str(id_f)]['bounding_box'][0]
            assigned_filters.append(filter_id)
        else:
            missed_id.append(filter_id)
    return filters, missed_id, id_rem, id_g, assigned, assigned_filters


def get_similarity_matrix(queries, galleries, query_ids, reminded_id, galleries_ids, assigned):
    matrix_sim = []
    for q_id, query in queries.items():
        if q_id in query_ids:
            sim, indices = calculate_similarity_faiss(query.visual_features, galleries)
            rows = [0] * len(reminded_id)
            for id in reminded_id:
                found_id = np.where(indices == galleries_ids.index(id))[0]
                rows[reminded_id.index(id)] = sim[found_id][0]
            matrix_sim.append(rows)
    print(matrix_sim)
    print(reminded_id)
    print(query_ids)
    matrix = np.array(matrix_sim)
    return matrix

def update_filter(missed_filters, measure, filters, id_f, id_m, first_matrix, missed_matrix, row, col, assigned, ids2remove1, ids2remove2):
    missed_filters[id_f].update(measure['position'][0])
    filters[id_f] = missed_filters[id_f]
    filters[id_f].visual_features = measure['visual_features'][0]
    filters[id_f].bounding = measure['bounding_box'][0]
    missed_matrix[row, :] = 0
    missed_matrix[:, col] = 0
    first_matrix[row, :] = 0
    first_matrix[:, col] = 0
    assigned.append(id_m)
    ids2remove1.append(id_m)
    ids2remove2.append(id_f)
    return filters, assigned, ids2remove1, ids2remove2, first_matrix, missed_matrix

def find_missed_id(filters, missed_filters, measurements, galleries,
                   assigned, id_rem, current_id, first_gallery, assigned_filters, threshold):
    id_d = missed_filters.keys()
    queries = {}
    id_missed = [int(key) for key in id_d]
    missed_gallery = []
    ids2remove1 = []
    ids2remove2 = []
    if len(missed_filters) > 0:
        id_g = list(range(len(first_gallery)))
        for i in id_missed:
            missed_gallery.append(missed_filters[i].visual_features)
        for id in id_rem:
            queries[id] = creat_new_filter(measurements[str(id)], id)
        first_matrix = get_similarity_matrix(queries, first_gallery, id_rem, id_missed, id_g, assigned)
        missed_matrix = get_similarity_matrix(queries, missed_gallery, id_rem, id_missed, id_missed, assigned)
        if len(first_matrix) > 0 and len(missed_matrix) > 0:
            while np.max(first_matrix) > threshold or np.max(missed_matrix) > threshold:
                if np.max(first_matrix) > np.max(missed_matrix):
                    max_index1 = np.argmax(first_matrix)
                    max_row1, max_col1 = np.unravel_index(max_index1, first_matrix.shape)
                    max_row2 = np.argmax(missed_matrix[:, max_col1])
                    max1 = first_matrix[max_row1, max_col1]
                    max2 = missed_matrix[max_row2, max_col1]
                else:
                    max_index1 = np.argmax(missed_matrix)
                    max_row1, max_col1 = np.unravel_index(max_index1, missed_matrix.shape)
                    max_row2 = np.argmax(first_matrix[:, max_col1])
                    max1 = missed_matrix[max_row1, max_col1]
                    max2 = first_matrix[max_row2, max_col1]

                if max1 >= max2:
                    filters, assigned, ids2remove1, ids2remove2, first_matrix, missed_matrix = update_filter(
                        missed_filters,measurements[str(id_rem[max_row1])], filters, id_missed[max_col1], id_rem[max_row1],
                        first_matrix, missed_matrix, max_row1, max_col1, assigned, ids2remove1, ids2remove2
                    )
                else:
                    filters, assigned, ids2remove1, ids2remove2, first_matrix, missed_matrix = update_filter(
                        missed_filters, measurements[str(id_rem[max_row2])], filters, id_missed[max_col1], id_rem[max_row2],
                        first_matrix, missed_matrix, max_row2, max_col1, assigned, ids2remove1, ids2remove2
                    )
        if len(ids2remove1) > 0:
            for id1 in ids2remove1:
                id_rem.remove(id1)
        if len(ids2remove2) > 0:
            for id2 in ids2remove2:
                if id2 in missed_filters:
                    id_missed.remove(id2)
                    missed_filters.pop(id2)
                    print('find id' + str(id2))
    return filters, missed_filters, assigned, assigned_filters, first_gallery, id_rem
