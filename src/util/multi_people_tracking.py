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

def list_generator(input_list):
    for item in input_list:
        yield item


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


def get_id(ids, attached, id_g):
    gen = list_generator(ids)
    found_id = -1
    print('ooooooooooooooooooooooooooooooooo')
    print(attached)
    if ids[0] not in attached:
        found_id = next(gen)
        print(found_id)
    else:
        while found_id in attached:
            found_id = next(gen)
            print(found_id)
    if -1 < found_id < len(id_g):
        found_id = id_g[found_id]
    else:
        found_id = -1
    print('oooooooooooooooooooooooooooooooo')
    return found_id


def find_tracker_newF(filter_i, positions, measurements, dt, galleries, attached, id_g):
    ids = []
    filter_i.predict(dt=dt)
    covariance_matrix = np.array([[1, 0], [0, 1]])
    neighbors = []
    if len(positions) > 0:
        neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix, id_g)
    if len(neighbors) > 0:
        gallery = []
        print(neighbors, id_g, measurements.keys())
        for neighbor in neighbors:
            gallery.append(measurements[str(neighbor)]['visual_features'][0])
        sim, indices = calculate_similarity_faiss(filter_i.visual_features, gallery)
        for index in indices:
            ids.append(neighbors[index])
        found_id = get_id(ids, attached, id_g)
        if sim[0] < 0.95:
            sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
            ids = [id_g[index] for index in indices if index > -1]
            found_id = get_id(ids, attached, id_g)
    else:
        sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
        ids = [id_g[index] for index in indices if index > -1]
        found_id = get_id(ids, attached, id_g)
    return found_id


def tracking(measurements, filters, frame_num, missed_filters, current_id):
    print('frame num', str(frame_num))
    print(measurements)
    if frame_num == 0:
        current_id = 0
        for i in range(0, len(measurements)):
            filter_i = creat_new_filter(measurements[str(i)], i)
            filters.append(filter_i)
            current_id += 1

    else:
        attached = []
        positions = []
        galleries = []
        id_d = measurements.keys()
        id_g = [int(key) for key in id_d]
        for i in id_g:
            positions.append(measurements[str(i)]['position'][0])
            galleries.append(measurements[str(i)]['visual_features'][0])
        for filter_i in filters:
            print('filter id'+str(filter_i.object_id))
            id_f = find_tracker_newF(filter_i, positions, measurements, dt, galleries, attached, id_g)
            print('idf'+str(id_f))
            if id_f > -1:
                attached.append(id_f)
                filter_i.update(measurements[str(id_f)]['position'][0])
                filter_i.visual_features = measurements[str(id_f)]['visual_features'][0]
                # print(filter_i.object_id,nearest_measurement)
                measurements.pop(str(id_f))
            else:
                missed_filters.append(filter_i)
                print(str(filter_i.object_id) + ' is missed in frame number' + str(frame_num))
                filters.pop(filter_i.object_id)

        ind = 0
        find_ids = []
        while ind < len(measurements):
            id_dic = measurements.keys()
            for k, mfilter_i in enumerate(missed_filters):
                id_f = find_tracker_newF(mfilter_i, positions, measurements, dt, galleries, attached, id_g)
                if id_f >= 0:
                    attached.append(id_f)
                    mfilter_i.update(measurements[str(id_f)]['position'][0])
                    mfilter_i.visual_features = measurements[str(id_f)]['visual_features'][0]
                    measurements.pop(str(id_f))
                    filters.append(mfilter_i)
                    # missed_filters.pop(k)
                    find_ids.append(id_f)
                    ind += 1
                    print('find person with id' + mfilter_i.object_id)
                else:
                    if len(id_dic) >= 0:
                        filter_i = creat_new_filter(measurements[list(id_dic)[ind]], current_id)
                        measurements.pop(str(list(id_dic)[ind]))
                        filters.append(filter_i)
                        print('new id' + str(current_id))
                        current_id += 1
        for i in find_ids:
            missed_filters.pop(i)
    print('frame number:' + str(frame_num))
    pp_data = []
    for filter_i in filters:
        print('ID:' + str(filter_i.object_id))
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

    return filters, missed_filters, current_id
