import os
import ast
import csv
import timm
from PIL import Image
from LATransformer.model import LATransformerTest
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from util.cube_projection import CubeProjection
from util.camera_info import get_info
from util.tracking_utils import *
from util.assign_pose2panoramic import *
import detect_people

model = detect_people.load_model()


def counter():
    num = 0
    while True:
        yield num
        num += 1


FACE_NAMES = ['back', 'front', 'left', 'right']
counter_gen = counter()
back_info, right_info, left_info, front_info = get_info()

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This will make sure no GPU is being used
device = "cpu"
batch_size = 8
gamma = 0.7
seed = 42

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)

# Create La-Transformer
model1 = LATransformerTest(vit_base, lmbd=8).to(device)

name = "la_with_lmbd_8"
save_path = os.path.join('Weights-20230803T150538Z-001/Weights/net_best.pth')
model1.load_state_dict(torch.load(save_path), strict=False)
print(model1.eval())

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
tracks = {}
current_object_id = 0
galleries = []
scan = []
with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/scan.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Read each row of the CSV file
    for row in reader:
        image_id = int(row[0])  # Extract the image ID from the first column
        ranges = [float(value) for value in row[1:]]  # Extract th
        scan.append(ranges)

dr_spaam = []
with open('/home/sepid/workspace/Thesis/GuidingRobot/data2/drspaam_data2.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Read each row of the CSV file
    for row in reader:
        dr_spaam.append(row)

data = {}
# for i in range(36, int(len(scan)/2)):
for i in range(50, 60):
    path = '/home/sepid/workspace/Thesis/GuidingRobot/data2/image_' + str(i) + '.jpg'
    print(path)
    dsides = {'back': {
        'bounding_boxes': [],
        'positions': []
    },
        'front': {
            'bounding_boxes': [],
            'positions': []
        },
        'left': {
            'bounding_boxes': [],
            'positions': []
        },
        'right': {
            'bounding_boxes': [],
            'positions': []
        }
    }
    if os.path.exists(path):
        img = cv2.imread(path)
        #             # print()
        back, left, right, front = laser_scan2xy(scan[i])
        points = []
        for d in dr_spaam[i]:
            dr_value = tuple_of_floats = ast.literal_eval(d)
            x = dr_value[0]
            y = dr_value[1]
            points.append((x, y))

        back_xy, left_xy, right_xy, front_xy = sides_points(points)
        img = Image.fromarray(img)
        sides = CubeProjection(img, '')
        sides.cube_projection()
        people = []
        people_img = []
        image = np.array(img)
        detected_org = detect_people.detect_person(image, model)
        for face, side_img in sides.sides.items():
            if face in FACE_NAMES:
                cv_image = np.array(side_img)
                detected = detect_people.detect_person(cv_image, model)
                print(face)
                print(detected)
                sorted_detected = sorted(detected, key=lambda x: x[0])
                print(sorted_detected)
                dsides[face]['bounding_boxes'] = sorted_detected
                if face == 'back':
                    pose = []

                    XY = selected_point(back_xy, back, back_info, face, sorted_detected, cv_image)
                    for xy in XY:
                        if xy[0] != 0 and xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                    dsides[face]['positions'] = pose
                    print('-------------------')
                elif face == 'front':
                    pose = []
                    XY = selected_point(front_xy, front, front_info, face, sorted_detected, cv_image)
                    for xy in XY:
                        if xy[0] != 0 and xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                    dsides['front']['positions'] = pose
                    print('-------------------')
                elif face == 'right':
                    pose = []
                    XY = selected_point(right_xy, right, right_info, face, sorted_detected, cv_image)
                    for xy in XY:
                        if xy[0] != 0 and xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                    dsides['right']['positions'] = pose
                    print('-------------------')
                elif face == 'left':
                    pose = []
                    XY = selected_point(left_xy, left, left_info, face, sorted_detected, cv_image)
                    for xy in XY:
                        if xy[0] != 0 and xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                    dsides['left']['positions'] = pose
                    print('-------------------')
        measurments, positions, galleries = assign_pose2panoramic(img, detected_org, dsides, model1)
        print(measurments)
        frame_num = next(counter_gen)
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
                filters.append(filter_i)
                filter_i.visual_features = measurments[str(i)]['visual_features'][0]
        else:
            # Predict the next state for each object
            ids = []
            attached = []
            # print(len(filters))
            for filter_i in filters:
                filter_i.predict(dt=dt)
                covariance_matrix = np.array([[1, 0], [0, 1]])
                neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)
                max = 0
                id = 0
                sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
                print(sim, indices)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                if len(neighbors) > 0:
                    gallary = []
                    for neighbor in neighbors:
                        gallary.append(measurments[str(neighbor)]['visual_features'][0])
                    sim, indices = calculate_similarity_faiss(filter_i.visual_features, gallary)
                    print(neighbors, 'dddffddddddddddddd')
                    print(sim, indices, 'kkkkkk')
                    if len(attached) > 0:
                        for i in range(len(indices)):
                            if neighbors[indices[i]] not in attached:
                                id = neighbors[indices[i]]
                                attached.append(id)
                                max = sim[i]
                                print(max)
                                break
                    else:
                        id = neighbors[indices[0]]
                        attached.append(id)
                        max = sim[0]
                        print(max)

                    max = sim[0]
                    if max < 0.95:
                        sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
                        print(sim, indices, 'nnnnnnnn')

                else:
                    sim, indices = calculate_similarity_faiss(filter_i.visual_features, galleries)
                    print(sim, indices, 'rrrrrrr')

                if len(attached) > 0:
                    for i in range(len(indices)):
                        if indices[i] not in attached:
                            id = indices[i]
                            attached.append(id)
                            max = sim[i]
                            print(max)
                            break
                else:
                    id = neighbors[indices[0]]
                    attached.append(id)
                    max = sim[0]
                    print(max)
                if max > 0.95:
                    print(filter_i.object_id, max, id)
                    filter_i.update(measurments[str(id)]['position'][0])
                    estimated_state = filter_i.x
                    estimated_covariance = filter_i.P
                    filter_i.visual_features = measurments[str(id)]['visual_features'][0]
                else:
                    print(str(filter_i.object_id) + 'is missed')
