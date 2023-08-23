#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm
import cv2

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as F
from cube_projection import CubeProjection
from PIL import Image
import pickle
import csv

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from tqdm import tqdm


def cartesian_to_polar(cartesian_points):

    x, y = cartesian_points[:, 0], cartesian_points[:, 1]
    angles = np.arctan2(y, x)  # Calculate angles in radians
    angles_deg = np.degrees(angles)  # Convert angles to degrees

    # Map angles to the [0°, 360°] range
    #angles_deg = (angles_deg + 360) % 360
    offset = -50
    for i in range(len(angles_deg)):
        if angles_deg[i] < 0:
            angles_deg[i] = 360 + angles_deg[i]
            offset = 50

    # Calculate magnitudes (distances)
    magnitudes = np.sqrt(x**2 + y**2)

    return np.column_stack((angles_deg, magnitudes))


def map_bounding_box_to_side(bounding_box):
    u1, y1, u2, v2 = bounding_box

    if (u1 >= 0 and u2 < 240) or (u1 >= 1720 and u2 < 1960):
        if 240 <= v2 < 720:
            return (0, "back")
    elif u1 >= 240 and u2 < 720:
        if 240 <= v2 < 720:
            return (1, "left")
    elif u1 >= 720 and u2 < 1200:
        if 240 <= v2 <720:
            return (2, "front")
    elif u1 >= 1200 and u2 < 1720:
        if 240 <= v2 < 720:
            return (3, "right")
    return None  # Bounding box doesn't fit any side


class ImagePointFinder:

    def __init__(self):

        ####### CAMERA MATRIX (TRIAL)
        calib_file = "../calibration_data_intrinsics/intrinsics.pkl"
        # Open the pickle file for reading in binary mode
        with open(calib_file, 'rb') as file:
            # Load the dictionary from the pickle file
            self.camera_calib = pickle.load(file)

        ############### PREPARE THE DETECTOR
        # Load the fine-tuned model's state dictionary
        sides = ["back", "left", "right", "front", "top", "bottom"]
        # create calibration folders
        for side in sides:
            if not os.path.exists(side):
                os.makedirs(side)
                print(f"Folder '{side}' created.")
            else:
                print(f"Folder '{side}' already exists.")

        model_state_dict = torch.load("one_shot_object_detector_5x3.pth")
        model_state_dict_side = torch.load("one_shot_object_detector_5x3_SIDES.pth")

        # Create an instance of the model
        self.model = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 1
        self.model.roi_heads.box_predictor.cls_score.out_features = num_classes
        self.model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
        self.model.load_state_dict(model_state_dict)

        # Send the model to CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        # Create an instance of the model
        self.model_side = fasterrcnn_resnet50_fpn(pretrained=False)
        num_classes = 1
        self.model_side.roi_heads.box_predictor.cls_score.out_features = num_classes
        self.model_side.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
        self.model_side.load_state_dict(model_state_dict_side)

        # Send the model to CUDA if available
        self.model_side.to(device)
        self.model_side.eval()

        ###################### PERFORM INFERENCE
        # Detect board with the trained model
        # folder_path = "images"
        # image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
        #                filename.endswith(('.jpg', '.png', '.jpeg'))]

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.device = device

    def processImage(self, image_path, verbose=True):

        img_id = image_path.split("/")[-1]

        print("Processing image: ", image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        test_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
        predictions = self.model(test_tensor)[0]

        # Find the index of the bounding box with the highest score
        # max_score_idx = predictions['scores'][:2]

        if len(predictions['boxes']) < 2:
            print("CHECKERBOARD NOT DETECT")
            return None

        # Get the bounding box and score with the highest score
        max_score_box = predictions['boxes'][:2].detach().cpu().numpy()
        # max_score = predictions['scores'][max_score_idx]

        # Calculate the minimum x, minimum y, maximum x, and maximum y values across all bounding boxes
        x1 = np.min(max_score_box[:, 0])
        y1 = np.min(max_score_box[:, 1])
        x2 = np.max(max_score_box[:, 2])
        y2 = np.max(max_score_box[:, 3])

        # Create the merged bounding box
        merged_bbox = [x1, y1, x2, y2]
        side = map_bounding_box_to_side(merged_bbox)

        if side is not None:
            cube = CubeProjection(Image.fromarray(image), ".")
            img_side = cube.cube_projection(face_id=side[0], img_id="cropped_{}".format(img_id))

            img_side = np.array(img_side)
            if verbose:
                plt.imshow(img_side)
                plt.show()

            test_tensor = F.to_tensor(img_side).unsqueeze(0).to(self.device)
            predictions = self.model_side(test_tensor)[0]

            # Find the index of the bounding box with the highest score
            # max_score_idx = predictions['scores'][:2]

            if len(predictions['boxes']) < 2:
                print("CHECKERBOARD NOT DETECT (2nd round)")
                return None

            # Get the bounding box and score with the highest score
            max_score_box_side = predictions['boxes'][:2].detach().cpu().numpy()

            if max_score_box_side[0, 0] < max_score_box_side[1, 0]:
                boxes = (max_score_box_side[0], max_score_box_side[1])
            else:
                boxes = (max_score_box_side[1], max_score_box_side[0])

            box_with_corners = []  # [left/right][img, cropped_image, box, corners2]

            for box in boxes:

                # Enlarge the bounding box by a factor of 1.2 (adjust as needed)
                enlargement_factor = 1.25
                box = [
                    int(np.round(
                        box[0] - (box[2] - box[0]) * (enlargement_factor - 1) / 2)),
                    int(np.round(
                        box[1] - (box[3] - box[1]) * (enlargement_factor - 1) / 2)),
                    int(np.round(
                        box[2] + (box[2] - box[0]) * (enlargement_factor - 1) / 2)),
                    int(np.round(
                        box[3] + (box[3] - box[1]) * (enlargement_factor - 1) / 2)),
                ]

                # Crop a slice of the image using the horizontal pixel coordinates
                cropped_image = img_side[box[1]:box[3], box[0]:box[2]]

                # Offset for the x and y coordinates
                offset_x = box[0]
                offset_y = box[1]

                # side = map_bounding_box_to_side(box)

                ######## CALIBRATION
                scale_factor = 2
                img = cv2.resize(cropped_image, [cropped_image.shape[1] * scale_factor,
                                                 cropped_image.shape[0] * scale_factor], cv2.INTER_CUBIC)

                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(gray, (5, 3), None)
                # If found, add object points, image points (after refining them)

                print(ret)
                if verbose:
                    plt.imshow(img)
                    plt.show()

                if ret == True:
                    # objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.criteria)
                    # imgpoints.append(corners2)
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (5, 3), corners2, ret)

                    offsets = np.zeros_like(corners2) + np.array((offset_x, offset_y))

                    box_with_corners.append((img, img_side, box, corners2, offsets))

            if len(box_with_corners) == 2:
                if verbose:
                    plt.subplot(1, 2, 1)
                    plt.imshow(box_with_corners[0][0])
                    plt.subplot(1, 2, 2)
                    plt.imshow(box_with_corners[1][0])
                    plt.show()

                corners = np.array([box_with_corners[0][3], box_with_corners[1][3]])
                offsets = np.array([box_with_corners[0][4], box_with_corners[1][4]])

                k = self.camera_calib[side[1]]['K']

                ##### SHIFT LEFT POINTS
                # Reshape the array to 2D (15x2)
                data = corners.reshape(-1, 2)
                offsets = offsets.reshape(-1, 2)
                # Sort the array by the first column (x values)
                sort_idx = data[:, 0].argsort()
                sorted_data = data[sort_idx]
                offsets = offsets[sort_idx]
                # Extract the first 3 points with the smallest x values
                left_x_points = (sorted_data[:5] / scale_factor) + offsets[:5]
                right_x_points = (sorted_data[-5:] / scale_factor) + offsets[-5:]

                if verbose:
                    plt.imshow(box_with_corners[0][1])
                    plt.scatter(left_x_points[:, 0], left_x_points[:, 1], color="blue")
                    plt.scatter(right_x_points[:, 0], right_x_points[:, 1], color="red")

                    plt.show()

                return [side, left_x_points, left_x_points]

            else:
                print("Unable to find corners, skipping...")
                return None

        else:
            print("Not possible to remap in a side, skipping...")
            return None



class LaserPointFinder:

    def __init__(self, laser_spec):

        template = [
            [0, 0],
            [0.415, 0],
            [0, 0.42],
            [0, 0.21],
            [0.21, 0],
            [0.315, 0],
            [0.213, 0],
            [0, 0.315],
            [0, 0.213]
        ]

        self.template = np.array(template)

        self.angle_max = laser_spec['angle_max']
        self.angle_min = laser_spec['angle_min']
        self.angle_increment = laser_spec['angle_increment']
        self.range_min = laser_spec['range_min']
        self.range_max = laser_spec['range_max']

        self.R = 0.420

        ##### PARAMETERS
        #################

        self.dist_BC = self.R * np.sqrt(2)
        self.BC_rescale = 0.75
        self.eps_point = 15
        self.num_point = 5
        self.aperture = 25  # degrees

    def processScan(self, ranges, verbose=True):

        ranges = np.array(ranges)
        mask = ranges <= self.range_max

        # Calculate the angles for each range measurement
        angles = np.arange(self.angle_min, self.angle_max, self.angle_increment)

        ranges = ranges[mask]
        angles = angles[mask]

        # Convert polar coordinates to Cartesian coordinates
        x = np.multiply(ranges, np.cos(angles))
        y = np.multiply(ranges, np.sin(angles))

        points = np.stack([x, y], axis=1)

        cont = 0
        selected_points = []

        # Plot the laser scan data
        if verbose:
            plt.scatter(points[:, 0], points[:, 1], c='orange', marker='x')
            plt.scatter(self.template[:, 0], self.template[:, 1], c='purple', marker='1')

        # Iterate through all combinations of 3 points
        for i in tqdm(range(len(points))):

            A = points[i]

            eps = abs(np.sin(self.angle_increment) * np.linalg.norm(A))
            eps = 2 * eps

            for j in range(len(points)):
                B = points[j]
                distance_AB = np.linalg.norm(B - A)  # Euclidean distance

                distance_A = np.linalg.norm(A)
                distance_B = np.linalg.norm(B)

                if abs(distance_AB - self.R) <= eps and distance_B < distance_A:

                    count_points_on_AB = 0  # Counter for points on line AB
                    count_points_on_AC = 0  # Counter for points on line AC

                    for k in range(len(points)):

                        C = points[k]

                        distance_AC = np.linalg.norm(C - A)  # Euclidean distance
                        distance_BC = np.linalg.norm(B - C)
                        distance_C = np.linalg.norm(C)

                        if distance_C < distance_A and abs(distance_AC - self.R) <= eps and abs(
                                distance_AC - distance_AB) <= eps and (distance_BC - self.dist_BC) <= eps:

                            vector_AB = B - A
                            vector_AC = C - A

                            cos_angle_AB = np.dot(vector_AB, vector_AC) / (
                                    np.linalg.norm(vector_AC) * np.linalg.norm(vector_AB))

                            cos_dist = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))

                            if (np.degrees(cos_dist) - 90) <= 0 and abs(np.degrees(cos_dist) - 90) < self.aperture:

                                # Count points on lines AB and AC

                                # Remove points A, B, and C from filtered_points
                                filtered_points = np.delete(points, [i, j, k], axis=0)

                                distances = np.sqrt(np.sum((filtered_points - A) ** 2, axis=1))
                                filtered_points = filtered_points[distances <= self.R]

                                for point in filtered_points:

                                    vector_AB = B - A
                                    vector_AC = C - A
                                    vector_AP = point - A

                                    # Calculate the cosine of the angle between vectors
                                    cos_angle_AB = np.dot(vector_AP, vector_AB) / (
                                            np.linalg.norm(vector_AP) * np.linalg.norm(vector_AB))
                                    cos_angle_AC = np.dot(vector_AP, vector_AC) / (
                                            np.linalg.norm(vector_AP) * np.linalg.norm(vector_AC))

                                    # Calculate the angular distances (angles in radians)
                                    distance_pAB = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))
                                    distance_pAC = np.arccos(np.clip(cos_angle_AC, -1.0, 1.0))

                                    if np.degrees(distance_pAB) <= self.eps_point:
                                        count_points_on_AB += 1
                                    if np.degrees(distance_pAC) <= self.eps_point:
                                        count_points_on_AC += 1

                                if count_points_on_AB > self.num_point and count_points_on_AC > self.num_point:

                                    if verbose:
                                        plt.scatter(A[0], A[1], c='cyan', marker='+', s=400, linewidths=3)
                                        plt.scatter(B[0], B[1], c='red', marker='+', s=400, linewidths=3)
                                        plt.scatter(C[0], C[1], c='blue', marker='+', s=400, linewidths=3)

                                    # print("+++++++ Found points A {}, B {}, C {} ++++++++++++++++++".format(A, B, C))
                                    #
                                    # print("distAB {}, distAC {}, distBC {}".format(distance_AB, distance_AC, distance_BC))
                                    # print("distA {}, distB {}, distC {}".format(distance_A, distance_B, distance_B))
                                    # print("distance_AC - R {} (<{}), distance_AB - R {} (<{}]".format(distance_AC - R, eps,
                                    #                                                                   distance_AB - R, eps))
                                    # print("distance_AC - distance_AB {} (<{}), distance_BC - dist_BC {} (>{})".format(
                                    #     distance_AC - distance_AB, eps, distance_BC, BC_rescale * dist_BC))
                                    # print("np.degrees(cos_dist) - 90 = {} (<{})".format(abs(np.degrees(cos_dist) - 90), 20))
                                    # print("pointsAB {} pointsAC {}".format(count_points_on_AB, count_points_on_AC))

                                    cont += 1
                                    selected_points.append({'A': A, "B": B, "C": C})

        print("NUMBER OF PATTERNS FOUND: ", cont)

        if cont == 0:
            return None

        if verbose:
            plt.xlabel('X Distance (m)')
            plt.ylabel('Y Distance (m)')
            plt.title('Laser Scan Data')
            plt.axis('equal')  # Equal aspect ratio
            plt.grid(True)

        final_point = []
        min_dist = 10000

        for center in selected_points:

            A = center['A']
            B = center['B']
            C = center['C']

            distances = np.sqrt(np.sum((points - A) ** 2, axis=1))
            window_points = points[distances <= 1.5 * self.R]

            theta = np.arctan2(B[1] - A[1], B[0] - A[0])

            # Apply the sliding window to the point cloud
            optimal_translation, optimal_rotation, result = self.template_matching(window_points, self.template,
                                                                              initial_params=np.array(
                                                                                  [A[0], A[1], theta]))

            # Calculate the distance from the result
            distance = result.fun

            if distance < min_dist:
                min_dist = distance
                final_point = (A, B, C)

        A, B, C = final_point

        print(final_point)
        print("MIN DIST: ", min_dist)

        if verbose:
            plt.scatter(A[0], A[1], c='cyan', marker='o', s=400, linewidths=3, label="Final prediction A", alpha=0.5)
            plt.scatter(B[0], B[1], c='red', marker='o', s=400, linewidths=3, label="Final prediction B", alpha=0.5)
            plt.scatter(C[0], C[1], c='blue', marker='o', s=400, linewidths=3, label="Final prediction C", alpha=0.5)

            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.legend()

            # Convert Cartesian to polar coordinates
            cartesian_points = np.array(final_point)
            polar_coordinates = cartesian_to_polar(cartesian_points)
            plt.title("DETECTION OF THE CORNERS IN THE LASER")
            plt.axis('equal')  # Equal aspect ratio
            plt.show()

        if min_dist < 0.180:
            return final_point
        else:
            print("Confidence is too low! Discard the detection...")
            return None

    def template_matching(self, point_cloud, template, initial_params=np.array([0, 0, 0])):
        # Define a function to minimize - the sum of distances
        def objective(params):
            translation = params[:2]
            rotation = params[2]

            # Rotate the template by the given angle
            rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                        [np.sin(rotation), np.cos(rotation)]])
            rotated_template = np.dot(template, rotation_matrix.T)

            # Translate the rotated template
            translated_template = rotated_template + translation

            # Calculate distances between translated template and point cloud
            distances = cdist(translated_template, point_cloud)

            # Return the negative sum of distances (to be minimized)
            return np.mean(distances)

        # Initial guess for parameters: [translation_x, translation_y, rotation]

        # Minimize the objective function
        result = minimize(objective, initial_params, method='Nelder-Mead')

        # Get the optimal translation and rotation values
        optimal_translation = result.x[:2]
        optimal_rotation = result.x[2]

        return optimal_translation, optimal_rotation, result


def main():

    verbose = True

    csv_file_path = './images/scan.csv'
    image_folder = './images'

    # Define laser specification
    laser_spec = {
        'frame_id': "base_link",
        'angle_min': -3.140000104904175,
        'angle_max': 3.140000104904175,
        'angle_increment': 0.005799999926239252,
        'range_min': 0.44999998807907104,
        'range_max': 25.0
    }

    img_processor = ImagePointFinder()
    laser_processor = LaserPointFinder(laser_spec)

    with open(csv_file_path, mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header row

        for row in csvreader:

            # cmd = input("Press Enter to publish the next data...    [press Q to stop]")
            # if cmd.lower() == 'q':
            #     print("EXITING...")
            #     break

            bag_name = row[0]
            #bag_name = "image_4" ###TESTING
            laser_ranges = [float(value) for value in row[1:]]

            #Prepare the related image
            image_filename = os.path.splitext(bag_name)[0] + '.png'
            image_path = os.path.join(image_folder, image_filename)

            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            #img_results = img_processor.processImage(image_path, verbose=False)
            #print(img_results)
            img_results = 1 #testing

            if img_results is not None:

                laser_results = laser_processor.processScan(laser_ranges, verbose=True)
                print(laser_results)


    print("************* PROCESSING ENDS")
























    # ####### CAMERA MATRIX (TRIAL)
    # calib_file = "../calibration_data_intrinsics/intrinsics.pkl"
    # # Open the pickle file for reading in binary mode
    # with open(calib_file, 'rb') as file:
    #     # Load the dictionary from the pickle file
    #     camera_calib = pickle.load(file)
    #
    # ############### PREPARE THE DETECTOR
    # # Load the fine-tuned model's state dictionary
    # sides = ["back", "left", "right", "front", "top", "bottom"]
    # # create calibration folders
    # for side in sides:
    #     if not os.path.exists(side):
    #         os.makedirs(side)
    #         print(f"Folder '{side}' created.")
    #     else:
    #         print(f"Folder '{side}' already exists.")
    #
    # model_state_dict = torch.load("one_shot_object_detector_5x3.pth")
    # model_state_dict_side = torch.load("one_shot_object_detector_5x3_SIDES.pth")
    #
    # # Create an instance of the model
    # model = fasterrcnn_resnet50_fpn(pretrained=False)
    # num_classes = 1
    # model.roi_heads.box_predictor.cls_score.out_features = num_classes
    # model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
    # model.load_state_dict(model_state_dict)
    #
    # # Send the model to CUDA if available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # model.eval()
    #
    # # Create an instance of the model
    # model_side = fasterrcnn_resnet50_fpn(pretrained=False)
    # num_classes = 1
    # model_side.roi_heads.box_predictor.cls_score.out_features = num_classes
    # model_side.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
    # model_side.load_state_dict(model_state_dict_side)
    #
    # # Send the model to CUDA if available
    # model_side.to(device)
    # model_side.eval()
    #
    # ###################### PERFORM INFERENCE
    # # Detect board with the trained model
    # folder_path = "images"
    # image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
    #                filename.endswith(('.jpg', '.png', '.jpeg'))]
    #
    # # termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #
    # for i, image_path in enumerate(image_paths):
    #     print("Processing image: ", image_path)
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     test_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    #     predictions = model(test_tensor)[0]
    #
    #     # Find the index of the bounding box with the highest score
    #     #max_score_idx = predictions['scores'][:2]
    #
    #     if len(predictions['boxes']) < 2:
    #         print("CHECKERBOARD NOT DETECT")
    #         continue
    #
    #     # Get the bounding box and score with the highest score
    #     max_score_box = predictions['boxes'][:2].detach().cpu().numpy()
    #     #max_score = predictions['scores'][max_score_idx]
    #
    #     # Calculate the minimum x, minimum y, maximum x, and maximum y values across all bounding boxes
    #     x1 = np.min(max_score_box[:, 0])
    #     y1 = np.min(max_score_box[:, 1])
    #     x2 = np.max(max_score_box[:, 2])
    #     y2 = np.max(max_score_box[:, 3])
    #
    #     # Create the merged bounding box
    #     merged_bbox = [x1, y1, x2, y2]
    #     side = map_bounding_box_to_side(merged_bbox)
    #
    #     if side is not None:
    #         cube = CubeProjection(Image.fromarray(image), ".")
    #         img_side = cube.cube_projection(face_id=side[0], img_id="int_cal_img{}".format(i))
    #
    #         img_side = np.array(img_side)
    #         plt.imshow(img_side)
    #         plt.show()
    #
    #         test_tensor = F.to_tensor(img_side).unsqueeze(0).to(device)
    #         predictions = model_side(test_tensor)[0]
    #
    #         # Find the index of the bounding box with the highest score
    #         # max_score_idx = predictions['scores'][:2]
    #
    #         if len(predictions['boxes']) < 2:
    #             print("CHECKERBOARD NOT DETECT (2nd round)")
    #             continue
    #
    #         # Get the bounding box and score with the highest score
    #         max_score_box_side = predictions['boxes'][:2].detach().cpu().numpy()
    #
    #         if max_score_box_side[0,0] < max_score_box_side[1,0]:
    #             boxes = ( max_score_box_side[0], max_score_box_side[1] )
    #         else:
    #             boxes = ( max_score_box_side[1], max_score_box_side[0] )
    #
    #         box_with_corners = [] # [left/right][img, cropped_image, box, corners2]
    #
    #         for box in boxes:
    #
    #             # Enlarge the bounding box by a factor of 1.2 (adjust as needed)
    #             enlargement_factor = 1.25
    #             box = [
    #                 int(np.round(
    #                     box[0] - (box[2] - box[0]) * (enlargement_factor - 1) / 2)),
    #                 int(np.round(
    #                     box[1] - (box[3] - box[1]) * (enlargement_factor - 1) / 2)),
    #                 int(np.round(
    #                     box[2] + (box[2] - box[0]) * (enlargement_factor - 1) / 2)),
    #                 int(np.round(
    #                     box[3] + (box[3] - box[1]) * (enlargement_factor - 1) / 2)),
    #             ]
    #
    #             # Crop a slice of the image using the horizontal pixel coordinates
    #             cropped_image = img_side[box[1]:box[3], box[0]:box[2]]
    #
    #             # Offset for the x and y coordinates
    #             offset_x = box[0]
    #             offset_y = box[1]
    #
    #             #side = map_bounding_box_to_side(box)
    #
    #             ######## CALIBRATION
    #             scale_factor = 2
    #             img = cv2.resize(cropped_image, [cropped_image.shape[1]*scale_factor,
    #                                              cropped_image.shape[0]*scale_factor], cv2.INTER_CUBIC)
    #
    #             gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #             # Find the chess board corners
    #             ret, corners = cv2.findChessboardCorners(gray, (5,3), None)
    #             # If found, add object points, image points (after refining them)
    #
    #             print(ret)
    #
    #             plt.imshow(img)
    #             plt.show()
    #
    #             if ret == True:
    #                 #objpoints.append(objp)
    #                 corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)
    #                 #imgpoints.append(corners2)
    #                 # Draw and display the corners
    #                 cv2.drawChessboardCorners(img, (5,3), corners2, ret)
    #
    #                 offsets = np.zeros_like(corners2) + np.array( (offset_x, offset_y))
    #
    #                 box_with_corners.append((img, img_side, box, corners2, offsets))
    #
    #         if len(box_with_corners) == 2:
    #
    #             plt.subplot(1, 2, 1)
    #             plt.imshow(box_with_corners[0][0])
    #             plt.subplot(1, 2, 2)
    #             plt.imshow(box_with_corners[1][0])
    #             plt.show()
    #
    #             corners = np.array([box_with_corners[0][3], box_with_corners[1][3]])
    #             offsets = np.array([box_with_corners[0][4], box_with_corners[1][4]])
    #
    #             k = camera_calib[side[1]]['K']
    #
    #             ##### SHIFT LEFT POINTS
    #             # Reshape the array to 2D (15x2)
    #             data = corners.reshape(-1, 2)
    #             offsets = offsets.reshape(-1, 2)
    #             # Sort the array by the first column (x values)
    #             sort_idx = data[:, 0].argsort()
    #             sorted_data = data[sort_idx]
    #             offsets = offsets[sort_idx]
    #             # Extract the first 3 points with the smallest x values
    #             left_x_points = (sorted_data[:5] / scale_factor) + offsets[:5]
    #             right_x_points = (sorted_data[-5:] / scale_factor) + offsets[-5:]
    #
    #             plt.imshow(box_with_corners[0][1])
    #             plt.scatter(left_x_points[:,0], left_x_points[:,1], color="blue")
    #             plt.scatter(right_x_points[:,0], right_x_points[:, 1], color="red")
    #
    #             plt.show()
    #
    #         else:
    #             print("Unable to find corners, skipping...")
    #
    #     else:
    #             print("Not possible to remap in a side, skipping...")



if __name__ == "__main__":

    main()