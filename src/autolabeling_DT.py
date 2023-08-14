import ast
import csv
import math
import os
from math import pi, atan2, hypot, floor

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from numpy import clip
from scipy.spatial.distance import cdist, mahalanobis
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from filterpy.kalman import UnscentedKalmanFilter
from scipy.spatial.distance import cdist, mahalanobis
from filterpy.kalman import MerweScaledSigmaPoints

import detect_people

model = detect_people.load_model()


class CubeProjection:
    def __init__(self, imgIn, output_path):
        self.output_path = output_path
        self.imagin = imgIn
        self.sides = {
            'back': None,
            'left': None,
            'front': None,
            'right': None,
            'top': None,
            'bottom': None
        }

    def cube_projection(self):
        imgIn = self.imagin
        inSize = imgIn.size
        faceSize = int(inSize[0] / 4)

        FACE_NAMES = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
            4: 'top',
            5: 'bottom'
        }

        for face in range(6):
            imgOut = Image.new('RGB', (faceSize, faceSize), 'black')
            self.convertFace(imgIn, imgOut, face)
            if self.output_path != '':
                imgOut.save(self.output_path + FACE_NAMES[face] + '.jpg')
            else:
                self.sides[FACE_NAMES[face]] = imgOut

    def outImg2XYZ(self, i, j, faceIdx, faceSize):
        a = 2.0 * float(i) / faceSize
        b = 2.0 * float(j) / faceSize

        if faceIdx == 0:  # back
            (x, y, z) = (-1.0, 1.0 - a, 1.0 - b)
        elif faceIdx == 1:  # left
            (x, y, z) = (a - 1.0, -1.0, 1.0 - b)
        elif faceIdx == 2:  # front
            (x, y, z) = (1.0, a - 1.0, 1.0 - b)
        elif faceIdx == 3:  # right
            (x, y, z) = (1.0 - a, 1.0, 1.0 - b)
        elif faceIdx == 4:  # top
            (x, y, z) = (b - 1.0, a - 1.0, 1.0)
        elif faceIdx == 5:  # bottom
            (x, y, z) = (1.0 - b, a - 1.0, -1.0)
        return (x, y, z)

    def convertFace(self, imgin, imgout, faceIdx):
        inSize = imgin.size
        print(inSize)
        outsize = imgout.size
        inpix = imgin.load()
        outpix = imgout.load()
        facesize = outsize[0]

        for xout in range(facesize):
            for yout in range(facesize):
                (x, y, z) = self.outImg2XYZ(xout, yout, faceIdx, facesize)
                theta = atan2(y, x)  # range -pi to pi
                r = hypot(x, y)
                phi = atan2(z, r)  # range -pi/2 to pi/2

                # source img coords
                uf = 0.5 * inSize[0] * (theta + pi) / pi
                vf = 0.5 * inSize[0] * (pi / 2 - phi) / pi

                # Use bilinear interpolation between the four surrounding pixels
                ui = floor(uf)  # coord of pixel to bottom left
                vi = floor(vf)
                u2 = ui + 1  # coords of pixel to top right
                v2 = vi + 1
                mu = uf - ui  # fraction of way across pixel
                nu = vf - vi

                # Pixel values of four corners
                A = inpix[int(ui % inSize[0]), int(clip(vi, 0, inSize[1] - 1))]
                B = inpix[int(u2 % inSize[0]), int(clip(vi, 0, inSize[1] - 1))]
                C = inpix[int(ui % inSize[0]), int(clip(v2, 0, inSize[1] - 1))]
                D = inpix[int(u2 % inSize[0]), int(clip(v2, 0, inSize[1] - 1))]

                # interpolate
                (r, g, b) = (
                    A[0] * (1 - mu) * (1 - nu) + B[0] * (mu) * (1 - nu) + C[0] * (1 - mu) * nu + D[0] * mu * nu,
                    A[1] * (1 - mu) * (1 - nu) + B[1] * (mu) * (1 - nu) + C[1] * (1 - mu) * nu + D[1] * mu * nu,
                    A[2] * (1 - mu) * (1 - nu) + B[2] * (mu) * (1 - nu) + C[2] * (1 - mu) * nu + D[2] * mu * nu)

                outpix[xout, yout] = (int(round(r)), int(round(g)), int(round(b)))


def counter():
    num = 0
    while True:
        yield num
        num += 1


def check_border(bounding_box):
    flag = False
    return flag


def find_closest_position(pose0, pose1):
    # Calculate distances for both positions
    distance_0 = distance(pose0)
    distance_1 = distance(pose1)

    if distance_0 < distance_1:
        return pose0
    else:
        return pose1


def assign_pose2panoramic(image, org_detected, sides_detected):
    print('++++++++++++++++++++++++++++++++++')
    print(org_detected)
    print(sides_detected)
    people_detected = {}
    sorted_people = sorted(org_detected, key=lambda x: x[0])
    print(sorted_people)
    j = 0
    sorted_positions = []
    print(len(sorted_people))
    while j < len(sorted_people):
        preprocced_image = load_and_preprocess_image(image, sorted_people[j])
        vect = extract_feature_single(model, preprocced_image, "cpu")
        vect_features = vect.view((-1)).numpy()
        if 0 <= sorted_people[j][0] < 240:
            bounding_boxes = sides_detected['back']['bounding_boxes']
            positions = sides_detected['back']['positions']
            if len(positions) > 0:
                if 0 <= sorted_people[j][2] < 240:
                    for bnd, pos in zip(bounding_boxes, positions):
                        if 0 <= bnd[0] < 480:
                            if 240 <= bnd[2] < 480:
                                people_detected[str(j)] = {
                                    'bounding_box': [],  # Initialize with an empty list
                                    'position': [],  # Initialize with default values, replace with actual values
                                    'visual_features': []
                                }
                                people_detected[str(j)]['visual_features'].append(vect_features)
                                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                                people_detected[str(j)]['position'].append(pos)
                                people_detected[str(j)]['bounding_box'].append(sorted_people.pop())
                                sorted_positions.append(pos)
                                j += 1
                        elif 240 <= bnd[0] < 480:
                            if 240 <= bnd[2] <= 480:
                                people_detected[str(j)] = {
                                    'bounding_box': [],  # Initialize with an empty list
                                    'position': [],  # Initialize with default values, replace with actual values
                                    'visual_features': []
                                }
                                people_detected[str(j)]['visual_features'].append(vect_features)
                                people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                                people_detected[str(j)]['position'].append(pos)
                                sorted_positions.append(pos)
                                j += 1
                elif 240 < sorted_people[j][2] < 720:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    if len(sides_detected['left']['positions']) > 0:
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        back_pos = sides_detected['back']['positions'][-1]
                        left_pos = sides_detected['left']['positions'][0]
                        pos = find_closest_position(back_pos, left_pos)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append(pos)
                        sorted_positions.append(pos)
                        j += 1
                    else:
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        pos = sides_detected['back']['positions'][-1]
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append(pos)
                        sorted_positions.append(pos)
                        j += 1
        elif 240 <= sorted_people[j][0] < 720:
            bounding_boxes = sides_detected['left']['bounding_boxes']
            positions = sides_detected['left']['positions']
            if len(positions) > 0:
                for bnd, pos in zip(bounding_boxes, positions):
                    if 240 <= sorted_people[j][2] < 720:
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append(pos)
                        sorted_positions.append(pos)
                        j += 1
                    elif 720 <= sorted_people[j][2] < 1200:
                        if len(sides_detected['front']['positions']) > 0:
                            left_pos = sides_detected['left']['positions'][-1]
                            front_pos = sides_detected['front']['positions'][0]
                            pos = find_closest_position(front_pos, left_pos)
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
                        else:
                            pos = sides_detected['left']['positions'][-1]
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
        elif 720 <= sorted_people[j][0] < 1200:
            bounding_boxes = sides_detected['front']['bounding_boxes']
            positions = sides_detected['front']['positions']
            if len(positions) > 0:
                for bnd, pos in zip(bounding_boxes, positions):
                    if 720 <= sorted_people[j][2] < 1200:
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append(pos)
                        sorted_positions.append(pos)
                        j += 1
                    elif 1200 <= sorted_people[j][2] < 1680:
                        if len(sides_detected['right']['positions']) > 0:
                            front_pos = sides_detected['front']['positions'][-1]
                            right_pos = sides_detected['right']['positions'][0]
                            pos = find_closest_position(front_pos, right_pos)
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
                        else:
                            pos = sides_detected['front']['positions'][-1]
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
        elif 1200 <= sorted_people[j][0] < 1680:
            bounding_boxes = sides_detected['right']['bounding_boxes']
            positions = sides_detected['right']['positions']
            if len(positions) > 0:
                for bnd, pos in zip(bounding_boxes, positions):
                    if 1200 <= sorted_people[j][2] < 1680:
                        people_detected[str(j)] = {
                            'bounding_box': [],  # Initialize with an empty list
                            'position': [],  # Initialize with default values, replace with actual values
                            'visual_features': []
                        }
                        people_detected[str(j)]['visual_features'].append(vect_features)
                        people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                        people_detected[str(j)]['position'].append(pos)
                        sorted_positions.append(pos)
                        j += 1
                    elif 1680 <= sorted_people[j][2] < 1920:
                        if len(sides_detected['back']['positions']) > 0:
                            right_pos = sides_detected['right']['positions'][-1]
                            back_pos = sides_detected['back']['positions'][0]
                            pos = find_closest_position(back_pos, right_pos)
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
                        else:
                            pos = sides_detected['right']['positions'][-1]
                            people_detected[str(j)] = {
                                'bounding_box': [],  # Initialize with an empty list
                                'position': [],  # Initialize with default values, replace with actual values
                                'visual_features': []
                            }
                            people_detected[str(j)]['visual_features'].append(vect_features)
                            people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                            people_detected[str(j)]['position'].append(pos)
                            sorted_positions.append(pos)
                            j += 1
        elif 1680 <= sorted_people[j][0] <= 1920:
            bounding_boxes = sides_detected['back']['bounding_boxes']
            positions = sides_detected['back']['positions']
            for bnd, pos in zip(bounding_boxes, positions):
                if 0 <= bnd[0] < 240:
                    people_detected[str(j)] = {
                        'bounding_box': [],  # Initialize with an empty list
                        'position': [],  # Initialize with default values, replace with actual values
                        'visual_features': []
                    }
                    people_detected[str(j)]['visual_features'].append(vect_features)
                    people_detected[str(j)]['bounding_box'].append(sorted_people[j])
                    people_detected[str(j)]['position'].append(pos)
                    sorted_positions.append(pos)
                    j += 1
    return people_detected, sorted_positions


def laser_scan2xy(msg):
    angle_min = -3.140000104904175
    angle_increment = 0.005799999926239252
    num_ranges = len(msg)
    xy_points = []

    for j in range(num_ranges):
        angle = angle_min + j * angle_increment
        r = float(msg[j])
        # converted_angle = math.degrees(angle)

        if not math.isinf(r) and r > 0.1:
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            xy_points.append((x, y))
    back, left, right, front = sides_points(xy_points)
    return back, left, right, front


def convert_robotF2imageF(tmpx, tmpy, side_info):
    H = side_info['H']
    fu = side_info['fu']
    fv = side_info['fv']
    u0 = side_info['u0']
    v0 = side_info['v0']
    Zc = H[4] * tmpx + H[5] * tmpy + H[8]
    u = ((fu * H[0] + u0 * H[4]) * tmpx + (fu * H[1] + u0 * H[5]) * tmpy + fu * H[6] + u0 * H[8]) / Zc
    v = ((fv * H[2] + v0 * H[4]) * tmpx + (fv * H[3] + v0 * H[5]) * tmpy + fv * H[7] + v0 * H[8]) / Zc
    return [u, v]


def check_intersection(d_bound, point, flag):
    if flag:
        offset = 15
    else:
        offset = 15
    if d_bound[0] < point[0] < d_bound[2]:
        if d_bound[1] < point[1] < d_bound[3]:
            return True
        elif abs(d_bound[3] - point[1]) <= offset:
            return True
    elif abs(d_bound[0] - point[0]) <= offset or abs(point[0] - d_bound[2]) <= offset:
        if d_bound[1] < point[1] < d_bound[3]:
            return True
        elif abs(d_bound[3] - point[1]) <= offset:
            return True
    return False


def distance(xy):
    # Calculate the Euclidean distance between two points
    return math.sqrt((xy[0] - 0) ** 2 + (xy[1] - 0) ** 2)


def check_points(x_y):
    minimum = (float('inf'), float('inf'))
    for xy in x_y:
        if distance(xy) < distance(minimum):
            minimum = xy
    return minimum


def check_xy(xy, face):
    x = xy[0]
    y = xy[1]
    if face == 'back':
        if xy[0] < 1.25 and abs(xy[1]) < 1.25:
            x = xy[0] + 0.5
    elif face == 'front':
        if abs(xy[0]) < 1.25 and abs(xy[1]) < 1.25:
            x = xy[0] - 0.3
    elif face == 'left':
        if abs(xy[0]) < 1.25 and abs(xy[1]) < 1.25:
            y = xy[1] - 1
    elif face == 'right':
        if abs(xy[0]) < 1.25 and abs(xy[1]) < 1.25:
            y = xy[1] + 1
    return x, y


def selected_point(side_xy, side, side_info, face, detected, side_img):
    XY_people = [(0, 0) for _ in range(len(detected))]
    for ind, person in enumerate(detected):
        flag = False
        p = []
        x_y = []
        for xy in side_xy:
            x, y = check_xy(xy, face)
            u, v = convert_robotF2imageF(x, y, side_info)
            # draw_circle_bndBOX(u, v, side_img)
            if u < 0:
                u = 0
            if v < 0:
                v = 0
            if check_intersection(person, (u, v), False):
                flag = True
                p.append((u, v))
                x_y.append((xy[0], xy[1]))

        x = 0
        y = 0
        # print('xy', x_y)
        if not flag:
            for xy in side:
                x, y = check_xy(xy, face)
                u, v = convert_robotF2imageF(x, y, side_info)
                #                 draw_circle_bndBOX(u, v, side_img)
                if v > 480:
                    v = 479
                if face == 'left':
                    u += 50
                    v -= 40
                if check_intersection(person, (u, v), True):
                    p.append((u, v))
                    x_y.append((xy[0], xy[1]))
        x = 0
        y = 0
        if len(p) > 1:
            for i, pr in enumerate(p):
                if pr in XY_people:
                    x_y.pop(i)
            # print(x_y)
            x, y = check_points(x_y)
        elif len(p) == 1:
            x, y = x_y[0]
        if 0 < abs(x) <= 7 and 0 < abs(x) <= 7:
            XY_people[ind] = (x, y)
            print(ind, person, (x, y))
    for xy in XY_people:
        x, y = check_xy(xy, face)
        u, v = convert_robotF2imageF(x, y, side_info)
    #         draw_circle_bndBOX(u, v, side_img)
    return XY_people


def draw_circle_bndBOX(u, v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)


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


def sides_points(dr_spaam):
    back_xy = []
    left_xy = []
    front_xy = []
    right_xy = []
    for d in dr_spaam:
        # dr_value = tuple_of_floats = ast.literal_eval(d)
        x = d[0]
        y = d[1]
        if y >= 0:
            if x > 0 and x >= y:
                back_xy.append((x, y))
            elif 0 < x < y:
                right_xy.append((x, y))
            elif x < 0 and abs(x) < y:
                right_xy.append((x, y))
            elif x < 0 and abs(x) >= y:
                front_xy.append((x, y))
        else:
            if x > 0 and x >= abs(y):
                back_xy.append((x, y))
            elif 0 < x < abs(y):
                left_xy.append((x, y))
            elif x < 0 and abs(x) < abs(y):
                left_xy.append((x, y))
            elif x < 0 and abs(x) >= abs(y):
                front_xy.append((x, y))
    return back_xy, left_xy, right_xy, front_xy


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def extract_feature_single(model, image, device="cpu"):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    return output[0].squeeze(dim=0).detach().cpu()


def load_and_preprocess_image(image, bounding_box):
    # Validate if the bounding_box is a list with 4 elements (left, upper, right, lower)
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise ValueError("Bounding box should be a list containing 4 elements (left, upper, right, lower).")
    print(bounding_box)
    # Crop the image based on the bounding box
    cropped_image = image.crop(bounding_box)

    transform_list = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)

    # Convert the cropped image to RGB and apply the transformations
    preprocessed_image = transform(cropped_image).unsqueeze(0)  # Add batch dimension

    return preprocessed_image


def load_and_preprocess_images(image1_path, image2_path):
    image1 = load_and_preprocess_image(image1_path)
    image2 = load_and_preprocess_image(image2_path)
    return image1, image2


def calculate_similarity(query_vector, gallery_vectors):
    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Normalize the gallery vectors
    gallery_vectors = gallery_vectors / np.linalg.norm(gallery_vectors)

    # Perform cosine similarity between query and gallery vectors
    similarity_scores = np.dot(query_vector, gallery_vectors.T)
    return similarity_scores


def global_nearest_neighbor(reference_points, query_points, covariance_matrix):
    distances = cdist(reference_points, query_points, lambda u, v: mahalanobis(u, v, covariance_matrix))
    neighbors = []
    for i, dis in enumerate(distances):
        if dis < 0.5:
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


if __name__ == '__main__':
    FACE_NAMES = ['back', 'front', 'left', 'right']
    counter_gen = counter()
    back_info = {
        'H': np.array([-1.3272, -7.0239, -0.13689, 0.43081, 7.0104, -1.2212, -0.047192, 8.2577, -0.77688]),
        'fu': 250.001420127782,
        'fv': 253.955300723887,
        'u0': 239.731339559399,
        'v0': 246.917074981568
    }

    right_info = {
        'H': np.array([1.3646, -0.33852, -0.18656, 0.21548, 0.26631, 1.3902, -0.2393, 1.1006, -0.037212]),
        'fu': 253.399373379354,
        'fv': 247.434371718165,
        'u0': 246.434570692999,
        'v0': 239.287976204900
    }

    left_info = {
        'H': np.array([0.15888, -0.036621, -0.021383, 0.025895, 0.030874, 0.16751, 0.035062, -0.16757, 0.002782]),
        'fu': 248.567135164434,
        'fv': 249.783014432268,
        'u0': 242.942149245269,
        'v0': 233.235264118894
    }

    front_info = {
        'H': np.array([-0.27263, -1.1756, 0.64677, -0.048135, 1.1741, -0.24661, -0.039707, -0.023353, -0.27371]),
        'fu': 239.720364104544,
        'fv': 242.389765646256,
        'u0': 237.571362200999,
        'v0': 245.039671395514
    }

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
    galleries = {}
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
    for i in range(170, 176):

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
            measurments, positions = assign_pose2panoramic(img, detected_org, dsides)
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
                    print(filter_i.object_id)
                    filter_i.predict(dt=dt)
                    covariance_matrix = np.array([[1, 0], [0, 1]])
                    neighbors = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)
                    max = 0
                    id = 0
                    for neighbor in neighbors:
                        # Calculate similarity scores between the query vector and gallery vectors
                        similarity_scores = calculate_similarity(filter_i.visual_features,
                                                                 measurments[str(i)]['visual_features'][0])
                        print(similarity_scores)
                        if similarity_scores > max:
                            max = similarity_scores
                            id = neighbor
                    print(max, id)
                    filter_i.update(measurments[str(id)]['positions'][0])
                    estimated_state = filter_i.x
                    estimated_covariance = filter_i.P

            #         positions = np.array(measurements)
            #         # Calculate distances between predicted state and frame positions
            #         # distances = np.linalg.norm([filter_i.x[:2]]-positions)
            #         # Find the index of the nearest neighbor
            #         covariance_matrix = np.array([[1, 0], [0, 1]])
            #         nearest_index = global_nearest_neighbor(positions, [filter_i.x[:2]], covariance_matrix)
            #
            #         if len(attached) == 0:
            #             nearest_measurement = np.array(positions[nearest_index]).reshape(num_measurements)
            #             attached.append(nearest_measurement)
            #             # print('a1', attached)
            #             # print(nearest_measurement)
            #             filter_i.predict(dt=dt)  # Pass the time step dt
            #             filter_i.update(nearest_measurement)
            #             # print(filter_i.object_id,nearest_measurement)
            #             estimated_state = filter_i.x  # Estimated state after each update
            #             estimated_covariance = filter_i.P
