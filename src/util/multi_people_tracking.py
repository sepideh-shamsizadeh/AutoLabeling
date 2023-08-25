import yaml

from src.util.tracking_utils import *
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


def get_id(ids, attached, id_g):
    gen = list_generator(ids)
    found_id = next(gen)
    if found_id in attached:
        try:
            while found_id in attached:
                found_id = next(gen)
        except Exception as e:
            found_id = -1
    if -1 < found_id < len(id_g):
        found_id = id_g[found_id]
    else:
        found_id = -1
    return found_id


def find_tracker_newF(filter_i, positions, measurements, dt, attached, id_g):
    ids = []
    filter_i.predict(dt=dt)
    neighbors = []
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
        found_id = get_id(ids, attached, id_g)
        if sim[0] < 0.9:
            found_id = -1
    return found_id


def check_neighbours(filters, measurements, id_rem, attached, positions, galleries, id_g):
    assigned_filters = []
    missed_id = []
    for filter_id, filter_i in filters.items():
        id_f = find_tracker_newF(filter_i, positions, measurements, dt, attached, id_g)
        if id_f > -1:
            attached.append(id_f)
            id_rem.remove(id_f)
            filter_i.update(measurements[str(id_f)]['position'][0])
            filter_i.visual_features = measurements[str(id_f)]['visual_features'][0]
            assigned_filters.append(filter_id)
    return filters, missed_id, id_rem, id_g, attached, assigned_filters

def check_all_others(filters, measurements, id_rem, attached, assigned_filters, missed_id, galleries, id_g):
    id_filters = []
    found_ids = []
    matrix_sim = []
    for filter_id, filter_i in filters.items():
        if filter_id not in assigned_filters:
            sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
            ids = [id_g[index] for index in indices if index > -1]
            found_id = get_id(ids, attached, id_g)
            rows = [0 for id_r in id_rem]
            if found_id > -1:
                rows[id_rem.index(found_id)] = sim[np.where(indices == found_id)[0]][0]
                matrix_sim.append(rows)
                id_filters.append(filter_i.object_id)
                found_ids.append(found_id)
            else:
                if filter_id not in missed_id:
                    missed_id.append(filter_id)
    matrix = np.array(matrix_sim)
    if len(id_filters) > 0:
        for id in id_rem:
            max_row_index = np.argmax(matrix[:, id_rem.index(id)])
            if id_filters[max_row_index] not in assigned_filters and matrix[max_row_index, id_rem.index(id)] > 0:
                attached.append(id)
                id_rem.remove(id)
                filters[id_filters[max_row_index]].update(measurements[str(id)]['position'][0])
                filters[id_filters[max_row_index]].visual_features = measurements[str(id)]['visual_features'][0]
                assigned_filters.append(id_filters[max_row_index])
    for filter_id, filter_i in filters.items():
        if filter_id not in assigned_filters:
            sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
            ids = [id_g[index] for index in indices if index > -1]
            id_f = get_id(ids, attached, id_g)
            if id_f > -1:
                attached.append(id_f)
                id_rem.remove(id_f)
                filter_i.update(measurements[str(id_f)]['position'][0])
                filter_i.visual_features = measurements[str(id_f)]['visual_features'][0]
                assigned_filters.append(filter_id)
            else:
                if filter_id not in missed_id:
                    missed_id.append(filter_id)
    return filters, missed_id, id_rem, id_g, attached, assigned_filters

def find_missed_id(filters, missed_filters, measurements,
                   attached, id_rem, current_id, first_gallery, assigned_filters):
    id_d = missed_filters.keys()
    id_missed = [int(key) for key in id_d]
    missed_gallery = []
    print(id_missed)
    id_g = list(range(len(first_gallery)))
    for i in id_missed:
        missed_gallery.append(missed_filters[i].visual_features)
        print(i, len(missed_gallery))
    for measure_id, measure in measurements.items():
        if int(measure_id) in id_rem:
            print(measure_id)
            sim, indices = calculate_similarity_faiss(measure['visual_features'][0], first_gallery)
            ids = [id_g[index] for index in indices if index > -1]
            id = get_id(ids, attached, id_g)
            print('fffff'+str(id))
            if id in id_missed:
                filter_i = creat_new_filter(measure, id)
                filters[id] = filter_i
                missed_filters.pop(id)
                print('find id' + str(id))
                attached.append(id)
                id_rem.remove(id)
            else:
                if len(missed_gallery) > 0:
                    sim, indices = calculate_similarity_faiss(measure['visual_features'][0], missed_gallery)
                    ids = [id_missed[index] for index in indices if index > -1]
                    id = get_id(ids, attached, id_g)
                    if id in id_missed:
                        filter_i = creat_new_filter(measure, id)
                        filters[id] = filter_i
                        missed_filters.pop(id)
                        print('find id' + str(id))
                        attached.append(id)
                        id_rem.remove(id)
                else:
                    filter_i = creat_new_filter(measure, current_id)
                    filters[current_id] = filter_i
                    current_id += 1
    return filters, missed_filters, attached, assigned_filters


def tracking(measurements, filters, frame_num, missed_filters, current_id, first_gallery):

    if frame_num == 0:
        first_gallery = []
        current_id = 0
        for i in range(0, len(measurements)):
            filter_i = creat_new_filter(measurements[str(i)], i)
            first_gallery.append(measurements[str(i)]['visual_features'][0])
            filters[current_id] = filter_i
            current_id += 1

    else:
        id_d = measurements.keys()
        id_g = [int(key) for key in id_d]
        id_rem = [int(key) for key in id_d]
        attached = []
        positions = []
        galleries = []
        for i in id_g:
            positions.append(measurements[str(i)]['position'][0])
            galleries.append(measurements[str(i)]['visual_features'][0])

        filters, missed_ids, id_rem, id_g, attached, assigned_filters = check_neighbours(
            filters, measurements, id_rem, attached, positions, galleries, id_g
        )

        filters, missed_ids, id_rem, id_g, attached, assigned_filters = check_all_others(
            filters, measurements, id_rem, attached, assigned_filters, missed_ids, galleries, id_g
        )

        filters, missed_filters = add_loss_of_id(filters, missed_ids, missed_filters)

        if len(id_rem) > 0:
            filters, missed_filters, attached, assigned_filters = find_missed_id(
                filters, missed_filters, measurements,
                attached, id_rem, current_id, first_gallery, assigned_filters
            )

    print('frame number:' + str(frame_num))

    pp_data = []
    for filter_id, filter_i in filters.items():
        print('ID:' + str(filter_id))
        print('Position:' + str(filter_i.x[:2]))
        position = {'x': float(filter_i.x[0]), 'y': float(filter_i.x[1])}
        pp = {'id ' + str(filter_i.object_id): position}
        print(pp)
        pp_data.append(pp)

    frame = 'frame ' + str(frame_num)
    yaml_data = {frame: pp_data}
    output_file = 'tracks.yaml'
    # Open the file in write mode
    with open(output_file, 'a') as file:
        yaml.dump(yaml_data, file)

    return filters, missed_filters, current_id, first_gallery
