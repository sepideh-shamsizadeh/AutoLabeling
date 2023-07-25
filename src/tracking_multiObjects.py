import yaml
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter
from scipy.spatial.distance import cdist, mahalanobis
from filterpy.kalman import MerweScaledSigmaPoints


def global_nearest_neighbor(reference_points, query_points, covariance_matrix):
    distances = cdist(reference_points, query_points, lambda u, v: mahalanobis(u, v, covariance_matrix))
    nearest_indices = np.argmin(distances)
    if distances[nearest_indices][0] < 0.7:
        return nearest_indices
    else:
        return -1

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

    # Perform any additional handling or cleanup for the lost ID
    # Here, you can add code to save or log the information about the lost track, perform any post-processing, etc.
    # For example:
    # print("Lost ID:", filter_i.object_id)
    # print("Final position:", filter_i.x[:2])
    return filters


# Set the parameters
num_states = 4  # Number of states (x, y, vx, vy)
num_measurements = 2  # Number of measurements (position)
process_noise_variance = 0.1  # Process noise variance
measurement_noise_variance = 0.01  # Measurement noise variance
dt = 0.01  # Time step
loss_association_threshold = 2  # Number of consecutive frames without association to consider loss of measurement association
removed_objects_p = []
removed_objects_i = []

# Define the process and measurement noise covariance matrices
Q = np.eye(num_states) * process_noise_variance
R = np.eye(num_measurements) * measurement_noise_variance

# Set initial state and covariance matrix
initial_covariance = np.eye(num_states) * 1.0  # Initial covariance matrix

# Create filters for each object
filters = []

with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/output0.yaml', 'r') as file:
    data = yaml.safe_load(file)

tracks = {}
current_object_id = 0
for frame, frame_data in data.items():
    print(frame)
    # Get measurements for the current frame
    measurements = []
    p_frames = []
    pp_data = []
    for person_data in frame_data:
        person_id = list(person_data.keys())[0]
        position = np.array([person_data[person_id]['x'], person_data[person_id]['y']])
        measurements.append(position)

    frame_tracks = {}
    if len(filters) == 0:
        for object_id, person in enumerate(measurements):
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
            filter_i.object_id = object_id
            current_object_id = object_id
            # print(filter_i.x[:])
            # print(filter_i.object_id)

            # Initialize loss of measurement association counter
            filter_i.loss_association_counter = 0
            filter_i.miss_frame = []
            filter_i.frame_num = float(frame.split(' ')[1])
            filters.append(filter_i)
    else:
        # Predict the next state for each object
        ids = []
        attached = []
        # print(len(filters))
        for filter_i in filters:
            positions = np.array(measurements)
            # Calculate distances between predicted state and frame positions
            # distances = np.linalg.norm([filter_i.x[:2]]-positions)
            # Find the index of the nearest neighbor
            covariance_matrix = np.array([[1, 0], [0, 1]])
            nearest_index = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)

            if len(attached) == 0:
                nearest_measurement = np.array(positions[nearest_index]).reshape(num_measurements)
                attached.append(nearest_measurement)
                # print('a1', attached)
                # print(nearest_measurement)
                filter_i.predict(dt=dt)  # Pass the time step dt
                filter_i.update(nearest_measurement)
                # print(filter_i.object_id,nearest_measurement)
                estimated_state = filter_i.x  # Estimated state after each update
                estimated_covariance = filter_i.P
            else:
                nearest_measurement = np.array(positions[nearest_index]).reshape(num_measurements)
                natt = True
                for a in attached:
                    if a[0]==nearest_measurement[0] and a[1]==nearest_measurement[1]:
                        natt = False
                if natt:

                    # print('ob', filter_i.object_id)
                    # print(filter_i.object_id)
                    # Update the state using the nearest neighbor measurement
                    nearest_measurement = np.array(positions[nearest_index]).reshape(num_measurements)
                    if filter_i.object_id == 1:
                        print(positions)
                        print(filter_i.x[:2])
                        print(nearest_measurement)
                    attached.append(nearest_measurement)
                    # print('a1', attached)
                    # print(nearest_measurement)
                    filter_i.predict(dt=dt)  # Pass the time step dt
                    filter_i.update(nearest_measurement)
                    # print(filter_i.object_id,nearest_measurement)
                    estimated_state = filter_i.x  # Estimated state after each update
                    estimated_covariance = filter_i.P
                    # print('ff',filter_i.x[:2])
                    # print("Track position:", estimated_state[:2])
                else:
                    # print('ii', filter_i.object_id)
                    filter_i.loss_association_counter += 1
                    filter_i.miss_frame.append(frame)
                    # Handle loss of ID and new ID assignments
        # print(frame, mes_seen)
        if len(measurements) > len(attached):
            not_in_attached = [element for element in measurements if all((element != arr).any() for arr in attached)]
            for ms in not_in_attached:
                if len(removed_objects_p)>0:
                    ms = np.array([[ms[0], ms[1]]])
                    ms_2d = ms.reshape(1, -1)  # Reshape ms to a 2D array with one row
                    positions = np.array(removed_objects_p)
                    # Calculate distances between predicted state and frame positions
                    covariance_matrix = np.array([[1, 0], [0, 1]])
                    nearest_index= global_nearest_neighbor(positions, [filter_i.x[:2]],
                                                                                 covariance_matrix)
                    if nearest_index > -1:
                        filter_i = UnscentedKalmanFilter(dim_x=num_states, dim_z=num_measurements, dt=dt,
                                                         fx=state_transition_fn, hx=measurement_fn,
                                                         points=MerweScaledSigmaPoints(num_states, alpha=0.1, beta=2.,
                                                                                       kappa=-1.0))
                        filter_i.x = [removed_objects_p[nearest_index][0], removed_objects_p[nearest_index][1], 0, 0]
                        filter_i.P = initial_covariance
                        filter_i.dim_x = num_states

                        # Set process and measurement noise covariance matrices
                        filter_i.Q = Q
                        filter_i.R = R

                        # Set object ID
                        filter_i.object_id = removed_objects_i[nearest_index]
                        # print('aga',filter_i.object_id)
                        # Initialize loss of measurement association counter
                        filter_i.loss_association_counter = 0
                        estimated_state = filter_i.x  # Estimated state after each update
                        estimated_covariance = filter_i.P
                        filter_i.miss_frame = []
                        filter_i.frame_num = float(frame.split(' ')[1])
                        filters.append(filter_i)
                        removed_objects_i.pop(nearest_index)
                        removed_objects_p.pop(nearest_index)

                else:
                    filter_i = UnscentedKalmanFilter(dim_x=num_states, dim_z=num_measurements, dt=dt,
                                                     fx=state_transition_fn, hx=measurement_fn,
                                                     points=MerweScaledSigmaPoints(num_states, alpha=0.1, beta=2.,
                                                                                   kappa=-1.0))

                    # Set initial state and covariance matrix
                    # print(frame, current_object_id, measurements[ind])
                    filter_i.x = [ms[0], ms[1], 0, 0]
                    filter_i.P = initial_covariance
                    filter_i.dim_x = num_states

                    # Set process and measurement noise covariance matrices
                    filter_i.Q = Q
                    filter_i.R = R

                    # Set object ID
                    current_object_id += 1
                    filter_i.object_id = current_object_id

                    # Initialize loss of measurement association counter
                    filter_i.loss_association_counter = 0
                    estimated_state = filter_i.x  # Estimated state after each update
                    estimated_covariance = filter_i.P
                    filter_i.miss_frame = []
                    filter_i.frame_num = float(frame.split(' ')[1])
                    filters.append(filter_i)
            # print("Track position:", estimated_state[:2])
    remove_filters = []
    for filter_i in filters:
        # print('check', filter_i.object_id)
        if filter_i.loss_association_counter >= loss_association_threshold:
            if abs(float(filter_i.miss_frame[loss_association_threshold-1].split(' ')[1]) - float(filter_i.miss_frame[loss_association_threshold-2].split(' ')[1]) )== 1:
                # print('loss', filter_i.object_id)
                # print('loss', frame)
                # Handle loss of ID
                # print(len(filters))
                removed_objects_p.append(filter_i.x[:2])
                removed_objects_i.append(filter_i.object_id)
                remove_filters.append(filter_i)
            else:
                position = {'x': float(filter_i.x[0]), 'y': float(filter_i.x[1])}
                pp = {'id' + str(filter_i.object_id): position}
                # print(pp)
                pp_data.append(pp)
        else:
            position = {'x': float(filter_i.x[0]), 'y': float(filter_i.x[1])}
            pp = {'id' + str(filter_i.object_id): position}
            # print(pp)
            pp_data.append(pp)
    yaml_data = {frame: pp_data}
    output_file = 'tracks1.yaml'

    # Open the file in write mode
    with open(output_file, 'a') as file:
        # Write the YAML data to the file
        yaml.dump(yaml_data, file)
    filters = handle_loss_of_id(filters, remove_filters)
    # print(removed_objects_i)
    print('-------------------------------------------------------------------------------------------')