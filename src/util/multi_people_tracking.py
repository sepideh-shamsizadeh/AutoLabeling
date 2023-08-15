from src.util.tracking_utils import *
from src.util.visual_feature_utils import *
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


def tracking(measurments, positions, galleries, filters, frame_num, missed_ids,
             loss_association_threshold, removed_objects_p, removed_objects_f, removed_objects_id):
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

    # Create filters for each object
    tracks = {}
    current_object_id = 0
    print(measurments)

    pp_data = []
    if frame_num == 0:
        for i in range(0, len(measurments)):
            person = measurments[str(i)]['position'][0]
            bounding_boxes = measurments[str(i)]['bounding_box'][0]
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
            filter_i.object_id = i
            current_object_id = i
            # print(filter_i.x[:])
            # print(filter_i.object_id)

            # Initialize loss of measurement association counter
            filter_i.loss_association_counter = 0
            filter_i.miss_frame = []
            filter_i.frame_num = frame_num
            filter_i.visual_features = measurments[str(i)]['visual_features'][0]
            filters.append(filter_i)
    else:
        # Predict the next state for each object
        attached = []
        # print(len(filters))
        for filter_i in filters:
            ids = []
            neighbor_flag = False
            filter_i.predict(dt=dt)
            covariance_matrix = np.array([[1, 0], [0, 1]])
            neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)
            if len(neighbors) > 0:
                gallary = []
                for neighbor in neighbors:
                    gallary.append(measurments[str(neighbor)]['visual_features'][0])
                sim, indices = calculate_similarity_faiss(filter_i.visual_features, gallary)
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
            if i > 0:
                attached.append(ids[i])
                filter_i.update(measurments[str(ids[i])]['position'][0])
                filter_i.visual_features = measurments[str(ids[i])]['visual_features'][0]
                # print(filter_i.object_id,nearest_measurement)
                estimated_state = filter_i.x  # Estimated state after each update
                estimated_covariance = filter_i.P
            else:
                print(str(filter_i.object_id)+' is missed')

    return filters, missed_ids, removed_objects_p, removed_objects_f, removed_objects_id
