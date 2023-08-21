import yaml

from src.util.tracking_utils import *
from src.util.visual_feature_utils import *
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


def creat_new_filter(measurement, id, initial_covariance, num_states, num_measurements, dt, Q, R):
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


def find_tracker_newF(filter_i, positions, measurements, dt, galleries, attached):
    ids = []
    neighbor_flag = False
    filter_i.predict(dt=dt)
    covariance_matrix = np.array([[1, 0], [0, 1]])
    neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)
    if len(neighbors) > 0:
        gallery = []
        for neighbor in neighbors:
            gallery.append(measurements[str(neighbor)]['visual_features'][0])
        sim, indices = calculate_similarity_faiss(filter_i.visual_features, gallery)
        for index in indices:
            ids.append(neighbors[index])
        if sim[0] < 0.95:
            sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
            ids = indices
        else:
            neighbor_flag = True
    else:
        sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
        ids = indices
    i = 0
    while ids[i] in attached:
        if sim[i] > 0.95:
            i += 1
        else:
            if neighbor_flag:
                sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
                ids = indices
            else:
                i = -1
                break
    return i, ids


def tracking(measurements, filters, frame_num, missed_filters, current_id):
    print('frame num', str(frame_num))
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
    print(measurements)
    positions = []
    galleries = []
    for p in range(0, current_id):
        if measurements[str(p)]:
            positions.append(measurements[str(p)]['position'][0])
            galleries.append(measurements[str(p)]['visual_features'][0])
        else:
            positions.append([-100, -100])
            galleries.append({-100, -100, -100, -100})
    if frame_num == 0:
        current_id = 0
        for i in range(0, len(measurements)):
            filter_i = creat_new_filter(measurements[str(i)], i, initial_covariance, num_states, num_measurements, dt,
                                        Q, R)
            filters.append(filter_i)
            current_id += 1

    else:
        attached = []
        ids = []
        for k, filter_i in enumerate(filters):
            i, ids = find_tracker_newF(filter_i, positions, measurements, dt, galleries, attached)
            if 0 <= i < len(ids):
                attached.append(ids[i])
                filter_i.update(measurements[str(ids[i])]['position'][0])
                filter_i.visual_features = measurements[str(ids[i])]['visual_features'][0]
                # print(filter_i.object_id,nearest_measurement)
                measurements.pop(str(ids[i]))
                estimated_state = filter_i.x  # Estimated state after each update
                estimated_covariance = filter_i.P

            else:
                missed_filters.append(filter_i)
                print(str(filter_i.object_id) + ' is missed in frame number' + str(frame_num))
                filters.pop(k)
        ind = 0
        find_ids = []
        while ind < len(measurements):
            id_dic = measurements.keys()
            for k, mfilter_i in enumerate(missed_filters):
                i, ids = find_tracker_newF(mfilter_i, positions, measurements, dt, galleries, attached)
                if i >= 0:
                    attached.append(ids[i])
                    mfilter_i.update(measurements[str(ids[i])]['position'][0])
                    mfilter_i.visual_features = measurements[str(ids[i])]['visual_features'][0]
                    # print(mfilter_i.object_id,nearest_measurement)
                    x = measurements.pop(ids[i])
                    print(x)
                    estimated_state = mfilter_i.x  # Estimated state after each update
                    estimated_covariance = mfilter_i.P
                    filters.append(mfilter_i)
                    # missed_filters.pop(k)
                    find_ids.append(k)
                    ind += 1
                    print('find person with id' + mfilter_i.object_id)
                else:
                    if len(id_dic):
                        filter_i = creat_new_filter(measurements[list(id_dic)[ind]], current_id, initial_covariance,
                                                    num_states,
                                                    num_measurements, dt, Q, R)
                        measurements.pop(list(id_dic)[ind])
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
