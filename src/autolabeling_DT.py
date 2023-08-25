import os
import ast
import csv
import timm
from LATransformer.model import LATransformerTest
from util.multi_people_tracking import tracking
from util.cube_projection import CubeProjection
from util.camera_info import get_info
from util.assign_pose2panoramic import *
import detect_people

detection_model = detect_people.load_model()


def counter():
    num = 0
    while True:
        yield num
        num += 1


FACE_NAMES = ['back', 'front', 'left', 'right']
back_info, right_info, left_info, front_info = get_info()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
batch_size = 8
gamma = 0.7
seed = 42

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)

# Create La-Transformer
feature_model = LATransformerTest(vit_base, lmbd=8).to(device)

name = "la_with_lmbd_8"
save_path = os.path.join('Weights-20230803T150538Z-001/Weights/net_best.pth')
feature_model.load_state_dict(torch.load(save_path), strict=False)
print(feature_model.eval())

counter_gen = counter()

scan = []
with open('../data/scan.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Read each row of the CSV file
    for row in reader:
        image_id = int(row[0])  # Extract the image ID from the first column
        ranges = [float(value) for value in row[1:]]  # Extract th
        scan.append(ranges)

dr_spaam = []
with open('../data/drspaam_data2.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.reader(file)
    # Read each row of the CSV file
    for row in reader:
        dr_spaam.append(row)

data = {}
loss_association_threshold = 2  # Number of consecutive frames without association to consider loss of measurement association
removed_objects_p = []
removed_objects_id = []
removed_objects_f = []
filters = {}
first_gallery = []
missed_filters = {}
missed_ids = []
# for i in range(36, int(len(scan)/2)):
current_id = 0
for i in range(0, len(dr_spaam)):
    path = '../data/image_' + str(i) + '.jpg'
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
        img1 = cv2.imread(path)
        back, left, right, front = laser_scan2xy(scan[i])
        points = []
        for d in dr_spaam[i]:
            dr_value = tuple_of_floats = ast.literal_eval(d)
            x = dr_value[0]
            y = dr_value[1]
            points.append((x, y))

        back_xy, left_xy, right_xy, front_xy = sides_points(points)
        # Convert BGR image to RGB
        cv2_image_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(cv2_image_rgb)
        # pil_image.show()

        sides = CubeProjection(pil_image, '')
        sides.cube_projection()
        people = []
        people_img = []
        detected_org, objects_poses = detect_people.detect_person(img1, detection_model)
        for face, side_img in sides.sides.items():
            if face in FACE_NAMES:
                cv_image = np.array(side_img)
                detected, objects_pose = detect_people.detect_person(cv_image, detection_model)
                sorted_detected = sorted(detected, key=lambda x: x[0])
                dsides[face]['bounding_boxes'] = sorted_detected
                if face == 'back':
                    pose = []

                    XY = selected_point(back_xy, back, back_info, face, sorted_detected, cv_image)
                    for kk, xy in enumerate(XY):
                        if xy[0] != 0 or xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                        else:
                            people.append((2.5, 0))
                            pose.append((2.5, 0))
                    dsides[face]['positions'] = pose
                elif face == 'front':
                    pose = []
                    XY = selected_point(front_xy, front, front_info, face, sorted_detected, cv_image)
                    for kk, xy in enumerate(XY):
                        if xy[0] != 0 or xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                        else:
                            people.append((-2.5, 0))
                            pose.append((-2.5, 0))
                    dsides['front']['positions'] = pose
                elif face == 'right':
                    pose = []
                    XY = selected_point(right_xy, right, right_info, face, sorted_detected, cv_image)
                    for kk, xy in enumerate(XY):
                        if xy[0] != 0 or xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                        else:
                            people.append((0, 2.5))
                            pose.append((0, 2.5))
                    dsides['right']['positions'] = pose
                elif face == 'left':
                    pose = []
                    XY = selected_point(left_xy, left, left_info, face, sorted_detected, cv_image)
                    for kk, xy in enumerate(XY):
                        if xy[0] != 0 or xy[1] != 0:
                            people.append((xy[0], xy[1]))
                            pose.append((xy[0], xy[1]))
                        else:
                            people.append((0, -2.5))
                            pose.append((0, -2.5))
                    dsides['left']['positions'] = pose
        measurements = assign_pose2panoramic(pil_image, detected_org, dsides, feature_model)
        frame_num = next(counter_gen)
        filters, missed_filters, current_id, first_gallery = tracking(
            measurements, filters, frame_num, missed_filters, current_id, first_gallery
        )
