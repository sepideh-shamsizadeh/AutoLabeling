#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import cv2

from read_calib_data import RangeImagePublisher
from sklearn.cluster import DBSCAN, MeanShift

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms.functional as F
import pickle
from scipy.spatial import ConvexHull

import torch.nn as nn
from trainLaserDetector import NeuralNetwork

import joblib
from sklearn.ensemble import RandomForestClassifier

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


def visualize_matching_result(point_cloud, template, optimal_translation, optimal_rotation):
    # Rotate the template by the optimal rotation
    rotation_matrix = np.array([[np.cos(optimal_rotation), -np.sin(optimal_rotation)],
                                [np.sin(optimal_rotation), np.cos(optimal_rotation)]])
    rotated_template = np.dot(template, rotation_matrix.T)

    # Translate the rotated template by the optimal translation
    translated_template = rotated_template + optimal_translation

    # Plotting
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], color='blue', label='Point Cloud')
    plt.scatter(translated_template[:, 0], translated_template[:, 1], color='red', label='Matched Template')

    # Highlight the matched points using a polygon
    polygon = Polygon(translated_template, fill=None, edgecolor='green', linewidth=2)
    plt.gca().add_patch(polygon)

    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Template Matching Result')
    plt.grid(True)
    plt.show()

def calculate_distances(point_cloud, reference_point):
    # Calculate Euclidean distances between each point in the point cloud and the reference point
    distances = np.sqrt(np.sum((point_cloud - reference_point)**2, axis=1))
    return distances
def template_matching(point_cloud, template, initial_params = np.array([0, 0, 0])):
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

def process2(ranges, image, laser_spec):

    ranges = np.array(ranges)
    R = 0.285
    dist_BC = R * np.sqrt(2)
    eps_point = 15
    num_point = 5
    aperture = 25 #degrees

    angle_max = laser_spec['angle_max']
    angle_min = laser_spec['angle_min']
    angle_increment = laser_spec['angle_increment']
    range_min = laser_spec['range_min']
    range_max = laser_spec['range_max']

    mask = ranges <= range_max

    # Calculate the angles for each range measurement
    angles = np.arange(angle_min, angle_max, angle_increment)

    ranges = ranges[mask]
    angles = angles[mask]

    # Convert polar coordinates to Cartesian coordinates
    x = np.multiply(ranges, np.cos(angles))
    y = np.multiply(ranges, np.sin(angles))

    points = np.stack([x, y], axis=1)

    if verbose:
        plt.scatter(points[:,0], points[:,1], c='red', marker='x')

    model_filename = 'laserpoint_detector.pth'
    # Define the model structure
    input_size = 6  # Adjust according to your model architecture
    model = NeuralNetwork(input_size)
    model.load_state_dict(torch.load(model_filename))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Move the model to the specified device

    #points = points[:100]

    X = []
    for i in tqdm(range(len(points))):

        A = points[i]

        eps = abs(np.sin(angle_increment) * np.linalg.norm(A))
        eps = 2*eps

        for j in range(len(points)):
            B = points[j]
            distance_AB = np.linalg.norm(B - A)  # Euclidean distance

            distance_A = np.linalg.norm(A)
            distance_B = np.linalg.norm(B)

            if abs(distance_AB - R) <= eps and distance_B < distance_A:

                count_points_on_AB = 0  # Counter for points on line AB
                count_points_on_AC = 0  # Counter for points on line AC

                for k in range(len(points)):

                    C = points[k]

                    distance_AC = np.linalg.norm(C - A)  # Euclidean distance
                    distance_BC = np.linalg.norm(B - C)
                    distance_C = np.linalg.norm(C)

                    if distance_C < distance_A and abs(distance_AC - R) <= eps and abs(
                            distance_AC - distance_AB) <= eps and (distance_BC - dist_BC) <= eps:

                        new_data = np.array([A, B, C]).reshape(-1, 6)

                        X.append(new_data)

    X = np.array(X).squeeze()
    print("Candidate points: ", X.shape)

    y_pred = model(torch.tensor(X, device=device, dtype=torch.float32))


    y_probs, pred = torch.max(y_pred, dim=1)
    pred = pred.detach().cpu().numpy()
    y_probs = y_probs.detach().cpu().numpy()
    labels = pred.astype(bool)

    filtered_points = X[labels].reshape(pred.sum(), 3, 2)
    probs_points = y_probs[labels]

    #best_fit = np.argmax(probs_points)
    #seleced_point = filtered_points[best_fit]

    # A = seleced_point[0,:]
    # B = seleced_point[1,:]
    # C = seleced_point[2,:]

    seleced_points = []

    for i,vertex in enumerate(filtered_points):

        A = vertex[0,:]
        B = vertex[1,:]
        C = vertex[2,:]

        count_points_on_AB = 0
        count_points_on_AC = 0

        for point in points:

            vector_AB = B - A
            vector_AC = C - A
            vector_AP = point - A

            distance_AB = np.linalg.norm(vector_AB)
            distance_AC = np.linalg.norm(vector_AC)
            distance_AP = np.linalg.norm(vector_AP)

            if distance_AP > distance_AB or distance_AP > distance_AC:
                continue

            # Calculate the cosine of the angle between vectors
            cos_angle_AB = np.dot(vector_AP, vector_AB) / (
                    np.linalg.norm(vector_AP) * np.linalg.norm(vector_AB))
            cos_angle_AC = np.dot(vector_AP, vector_AC) / (
                    np.linalg.norm(vector_AP) * np.linalg.norm(vector_AC))

            # Calculate the angular distances (angles in radians)
            distance_pAB = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))
            distance_pAC = np.arccos(np.clip(cos_angle_AC, -1.0, 1.0))

            if np.degrees(distance_pAB) <= eps_point:
                count_points_on_AB += 1
            if np.degrees(distance_pAC) <= eps_point:
                count_points_on_AC += 1

        # if i == 19:
        #     plt.scatter(points[:, 0], points[:, 1], c='red', marker='x')
        #     plt.scatter(A[0], A[1], c='cyan', marker='o', s=100, linewidths=3, label="A", alpha=0.25)
        #     plt.scatter(B[0], B[1], c='orange', marker='o', s=100, linewidths=3, label="B", alpha=0.25)
        #     plt.scatter(C[0], C[1], c='green', marker='o', s=100, linewidths=3, label="C", alpha=0.25)
        #     print("count_points_on_AB: {} - count_points_on_AC: {}".format(count_points_on_AB, count_points_on_AC))
        #     plt.show()

        if count_points_on_AB > num_point and count_points_on_AC > num_point:
            seleced_points.append(vertex)


    seleced_points = np.array(seleced_points)
    print("Selected points: ", seleced_points.shape)

    A = seleced_points[:,0,:]
    B = seleced_points[:,1,:]
    C = seleced_points[:,2,:]

    if verbose:
        # plt.scatter(A[0], A[1], c='cyan', marker='o', s=100, linewidths=3, label="A", alpha=0.25)
        # plt.scatter(B[0], B[1], c='orange', marker='o', s=100, linewidths=3, label="B", alpha=0.25)
        # plt.scatter(C[0], C[1], c='green', marker='o', s=100, linewidths=3, label="C", alpha=0.25)

        plt.scatter(A[:,0], A[:,1], c='cyan', marker='o', s=100, linewidths=3, label="A", alpha=0.25)
        plt.scatter(B[:,0], B[:,1], c='orange', marker='o', s=100, linewidths=3, label="B", alpha=0.25)
        plt.scatter(C[:,0], C[:,1], c='green', marker='o', s=100, linewidths=3, label="C", alpha=0.25)

        plt.show()

def process(ranges, image, laser_spec):

    template = [
        [0,0],
        [0.285, 0],
        [0, 0.285],
        [0, 0.142],
        [0.142, 0],
        [0.07, 0],
        [0.213, 0],
        [0, 0.07],
        [0, 0.213]
    ]

    template = np.array(template)
    ranges = np.array(ranges)

    angle_max = laser_spec['angle_max']
    angle_min = laser_spec['angle_min']
    angle_increment = laser_spec['angle_increment']
    range_min = laser_spec['range_min']
    range_max = laser_spec['range_max']

    mask = ranges <= range_max

    # Calculate the angles for each range measurement
    angles = np.arange(angle_min, angle_max, angle_increment)

    ranges = ranges[mask]
    angles = angles[mask]

    # Convert polar coordinates to Cartesian coordinates
    x = np.multiply(ranges, np.cos(angles))
    y = np.multiply(ranges, np.sin(angles))

    points = np.stack([x,y], axis=1)

    R = 0.285

    ##### PARAMETERS
    #################

    dist_BC = R * np.sqrt(2)
    BC_rescale = 0.75
    eps_point = 15
    num_point = 5
    aperture = 25 #degrees

    cont = 0
    selected_points = []

    # Plot the laser scan data
    if verbose:
        plt.scatter(points[:,0], points[:,1], c='orange', marker='x')
        plt.scatter(template[:,0], template[:,1], c='purple', marker='1')

    # Iterate through all combinations of 3 points
    for i in tqdm(range(len(points))):

        A = points[i]

        eps = abs(np.sin(angle_increment) * np.linalg.norm(A))
        eps = 2*eps

        for j in range(len(points)):
            B = points[j]
            distance_AB = np.linalg.norm(B - A)  # Euclidean distance

            distance_A = np.linalg.norm(A)
            distance_B = np.linalg.norm(B)

            if abs(distance_AB - R) <= eps and distance_B < distance_A:

                count_points_on_AB = 0  # Counter for points on line AB
                count_points_on_AC = 0  # Counter for points on line AC

                for k in range(len(points)):

                    C = points[k]

                    distance_AC = np.linalg.norm(C - A)  # Euclidean distance
                    distance_BC = np.linalg.norm(B - C)
                    distance_C = np.linalg.norm(C)

                    if distance_C < distance_A and abs(distance_AC - R) <= eps and abs(distance_AC - distance_AB)<=eps and (distance_BC - dist_BC) <= eps :

                        vector_AB = B - A
                        vector_AC = C - A

                        cos_angle_AB = np.dot(vector_AB, vector_AC) / (
                                np.linalg.norm(vector_AC) * np.linalg.norm(vector_AB))

                        cos_dist = np.arccos(np.clip(cos_angle_AB, -1.0, 1.0))

                        if (np.degrees(cos_dist) - 90)<=0 and abs(np.degrees(cos_dist) - 90) < aperture:

                            #Count points on lines AB and AC

                            # Remove points A, B, and C from filtered_points
                            filtered_points = np.delete(points, [i, j, k], axis=0)

                            distances = np.sqrt(np.sum((filtered_points - A) ** 2, axis=1))
                            filtered_points = filtered_points[distances <= R]

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

                                if np.degrees(distance_pAB) <= eps_point:
                                    count_points_on_AB += 1
                                if np.degrees(distance_pAC) <= eps_point:
                                    count_points_on_AC += 1

                            if count_points_on_AB > num_point and count_points_on_AC > num_point:

                                if verbose:
                                    plt.scatter(A[0], A[1], c='cyan', marker='+', s=400, linewidths=3)
                                    plt.scatter(B[0], B[1], c='red', marker='+', s=400,  linewidths=3)
                                    plt.scatter(C[0], C[1], c='blue', marker='+', s=400,  linewidths=3)

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
                                selected_points.append({'A':A, "B":B, "C":C})

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
        window_points = points[distances <= 1.5*R]

        theta = np.arctan2(B[1]-A[1],B[0]-A[0])

        # Apply the sliding window to the point cloud
        optimal_translation, optimal_rotation, result = template_matching(window_points, template, initial_params=np.array([A[0], A[1], theta]))

        # Calculate the distance from the result
        distance = result.fun

        if distance < min_dist:
            min_dist = distance
            final_point = (A, B, C)


    A,B,C = final_point

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


        plt.show()

    if min_dist < 0.180:
        return final_point
    else:
        print("Confidence is too low! Discard the detection...")
        return None

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


def map_polar_to_horizontal_pixel(polar_points, image_width):
    angle_per_pixel = 360.0 / image_width

    horizontal_pixel_coords = []
    for angle_deg, _ in polar_points:
        # Calculate horizontal pixel coordinate
        pixel_coord = image_width - int(angle_deg / angle_per_pixel)
        horizontal_pixel_coords.append(pixel_coord)

    return horizontal_pixel_coords

def are_angles_close(angle1, angle2, threshold):
    angle_difference = np.abs(angle1 - angle2)
    return angle_difference < threshold or angle_difference > np.pi - threshold


def process_image(cartesian_points, image , detector):

    ############################################################################################

    # Access the backbone model (ResNet-50 in this case)
    backbone_model = detector.backbone
    # Check the device of the backbone model
    device = next(backbone_model.parameters()).device

    #Detect board with the trained model
    test_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    predictions = detector(test_tensor)[0]

    # Find the index of the bounding box with the highest score
    if len(predictions['scores']) == 0:
        print("**** Unable to detect the board!")
        return None

    max_score_idx = torch.argmax(predictions['scores'])

    # Get the bounding box and score with the highest score
    max_score_box = predictions['boxes'][max_score_idx].detach().cpu().numpy()
    max_score = predictions['scores'][max_score_idx]

    side = map_bounding_box_to_side(max_score_box)

    # Enlarge the bounding box by a factor of 1.2 (adjust as needed)
    enlargement_factor = 1.2

    # Calculate the new coordinates after applying the enlargement factor
    new_x1 = int(np.round(max_score_box[0] - (max_score_box[2] - max_score_box[0]) * (enlargement_factor - 1) / 2))
    new_y1 = int(np.round(max_score_box[1] - (max_score_box[3] - max_score_box[1]) * (enlargement_factor - 1) / 2))
    new_x2 = int(np.round(max_score_box[2] + (max_score_box[2] - max_score_box[0]) * (enlargement_factor - 1) / 2))
    new_y2 = int(np.round(max_score_box[3] + (max_score_box[3] - max_score_box[1]) * (enlargement_factor - 1) / 2))

    # Ensure the new coordinates are within [0, img_size]
    h, w, c = image.shape  # Replace with your actual image size
    new_x1 = max(0, min(new_x1, w - 1))
    new_y1 = max(0, min(new_y1, h - 1))
    new_x2 = max(0, min(new_x2, w - 1))
    new_y2 = max(0, min(new_y2, h - 1))

    # The adjusted coordinates are now in [0, img_size]
    box = [new_x1, new_y1, new_x2, new_y2]

    print("box: ", box)
    #############################################################################################

    # Crop a slice of the image using the horizontal pixel coordinates
    cropped_image = image[box[1]:box[3], box[0]:box[2]]

#    plt.imshow(cropped_image)
#    plt.show()

    # Resize the cropped image to have a width of 250 pixels
    new_width = 250
    height, width, _ = cropped_image.shape
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    cropped_image = cv2.resize(cropped_image, (new_width, new_height))

    # Convert the cropped region to grayscale
    gray_cropped_region = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection (e.g., Canny) to the grayscale image
    edges = cv2.Canny(gray_cropped_region, threshold1=50, threshold2=100, apertureSize=3)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, rho=3, theta=np.pi / 180, threshold=80)

    ################################# VISUALIZATION

    # Draw the detected lines on the cropped image
    line_image = np.copy(cropped_image)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red

    # Convert BGR to RGB for visualization with matplotlib
    line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

    # Display the edge image and the cropped image with detected lines
    if verbose:
        plt.figure(figsize=(18, 9))

        plt.subplot(1, 2, 1)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(line_image_rgb)
        plt.title('Cropped Image with Detected Lines')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    ###############################################

    # Merge lines with similar orientations and locations by averaging
    clustered_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            x0 = np.cos(theta) * rho
            y0 = np.sin(theta) * rho

            # Check if the line can be merged with any existing cluster
            merged = False
            for cluster in clustered_lines:
                cluster_rho, cluster_theta = cluster[0]
                # cluster_x0 = np.cos(cluster_theta) * cluster_rho
                # cluster_y0 = np.sin(cluster_theta) * cluster_rho

                # Compare angles with consideration of circular nature
                angle_difference = np.arctan2(np.sin(theta - cluster_theta), np.cos(theta - cluster_theta))

                if (np.abs(angle_difference) < np.pi / 8 or np.abs(angle_difference-np.pi) < np.pi / 10):
                    if abs(abs(rho) - abs(cluster_rho)) < 50:
                        cluster.append((rho, theta))
                        merged = True
                        break

            if not merged:
                clustered_lines.append([(rho, theta)])

    # Merge each cluster by averaging the lines inside it
    merged_lines = []
    for cluster in clustered_lines:
        cluster_rhos, cluster_thetas = zip(*cluster)
        avg_rho = np.mean(cluster_rhos)
        avg_theta = np.mean(cluster_thetas)
        merged_lines.append((avg_rho, avg_theta))


    ################################# VISUALIZATION

    # Visualize the lines on the cropped region
    if verbose:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(line_image)
        plt.title('Cropped Image')
        plt.axis('off')

    # Plot the detected lines
    line_image = np.copy(cropped_image)
    for line in merged_lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw lines in red

        if verbose:
            plt.subplot(1, 2, 2)
            plt.imshow(line_image)
            plt.title('Edges')
            plt.axis('off')

    if verbose:
        plt.title('Detected Lines')
        plt.axis('off')
        plt.show()

##################################

    # Find line intersections
    intersections = []
    for i in range(len(merged_lines)):
        rho1, theta1 = merged_lines[i]
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)

        for j in range(i + 1, len(merged_lines)):
            rho2, theta2 = merged_lines[j]
            a2 = np.cos(theta2)
            b2 = np.sin(theta2)

            det = a1 * b2 - a2 * b1
            if det != 0:  # Lines are not parallel
                x = int((b2 * rho1 - b1 * rho2) / det)
                y = int((a1 * rho2 - a2 * rho1) / det)
                intersections.append((x, y))

    # Convert intersections to a numpy array
    intersection_points = np.array(intersections)

    if len(intersection_points) == 0:
        print("No intersections found!")
        return None

    # Define the cropped bounding box's top-left corner coordinates
    cropped_top_left = (box[0], box[1])
    # Calculate the scale factor used for resizing the cropped image
    scale_factor = new_width / (box[2] - box[0])
    # Initialize an array to store remapped corners in the original image
    remapped_corners = []

    # Loop through each detected corner
    for corner in intersection_points:
        # Adjust the corner coordinates for the resizing and cropped bounding box's top-left corner
        original_corner = np.array([
            (corner[0] / scale_factor) + cropped_top_left[0],
            (corner[1] / scale_factor) + cropped_top_left[1]
        ])

        # Append the remapped corner to the array
        remapped_corners.append(original_corner)

    remapped_corners = np.array((remapped_corners))

    filtered_intersections = []

    for x, y in remapped_corners:
        if max_score_box[0] <= x <= max_score_box[2] and max_score_box[1] <= y <= max_score_box[3]:
            filtered_intersections.append((x, y))

    # Convert filtered intersections to a numpy array
    filtered_intersection_points = np.array(filtered_intersections)

    if len(filtered_intersection_points) == 0:
        print("No intersections in the ROI found!")
        return None

    cluster_centers = filtered_intersection_points

    # Define the thresholds for classification
    boundary_threshold = 0.0  # Adjust this value based on your requirements
    boundary_threshold2 = 1.0  # Adjust this value based on your requirements
    center_threshold = 0.5  # Adjust this value based on your requirements

    # Initialize lists to store the categorized points
    points_category_A = []
    points_category_B = []
    points_category_C = []

    # Iterate through each cluster center
    for point in cluster_centers:
        # Calculate the distance of the point's y-coordinate from the center of the max_score_box
        distance_from_center = point[0] - max_score_box[0]

        # Calculate the relative position of the point within the box
        relative_position = distance_from_center / (max_score_box[2] - max_score_box[0])

        if abs(relative_position-boundary_threshold) <= 0.1:
            points_category_B.append(point)
        elif abs(relative_position-center_threshold) <= 0.1:
            points_category_A.append(point)
        elif abs(relative_position - boundary_threshold2) <= 0.1:
            points_category_C.append(point)

    # Convert the categorized points lists to numpy arrays
    points_category_A = np.array(points_category_A)
    points_category_B = np.array(points_category_B)
    points_category_C = np.array(points_category_C)


    ###################################################################################

    # Define the bandwidth for Mean Shift clustering
    bandwidth = 5  # Adjust the bandwidth as needed

    # Initialize MeanShift clustering
    meanshift = MeanShift(bandwidth=bandwidth)

    # Perform MeanShift clustering and get cluster labels for each category
    cluster_centers_A = []
    if len(points_category_A) > 0:
        labels_A = meanshift.fit_predict(points_category_A)
        unique_labels_A = np.unique(labels_A)
        for label in unique_labels_A:
            cluster_points = points_category_A[labels_A == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers_A.append(cluster_center)

    cluster_centers_B = []
    if len(points_category_B) > 0:
        labels_B = meanshift.fit_predict(points_category_B)
        unique_labels_B = np.unique(labels_B)
        for label in unique_labels_B:
            cluster_points = points_category_B[labels_B == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers_B.append(cluster_center)

    cluster_centers_C = []
    if len(points_category_C) > 0:
        labels_C = meanshift.fit_predict(points_category_C)
        unique_labels_C = np.unique(labels_C)
        for label in unique_labels_C:
            cluster_points = points_category_C[labels_C == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers_C.append(cluster_center)


    points_category_A = np.array(cluster_centers_A)
    points_category_B = np.array(cluster_centers_B)
    points_category_C = np.array(cluster_centers_C)

    ########################################################à

    # Draw the detected lines on the cropped region
    result = cropped_image.copy()

    if merged_lines is not None:
        for rho, theta in merged_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), [255,0,0], 2)


    # Display the result
    if verbose:
        plt.title("CORNERS IN THE IMAGE")
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image)

        plt.scatter( filtered_intersection_points[:,0] ,  filtered_intersection_points[:,1], marker="x", s=100, color='orange', linewidths=2 )
        if len(points_category_A)>0:
            plt.scatter( points_category_A[:,0] ,  points_category_A[:,1], marker="x", s=100, color='red', linewidths=2, label="A" )
        if len(points_category_B) > 0:
            plt.scatter( points_category_B[:,0] ,  points_category_B[:,1], marker="x", s=100, color='blue', linewidths=2, label="B")
        if len(points_category_C)>0:
            plt.scatter( points_category_C[:,0] ,  points_category_C[:,1], marker="x", s=100, color='cyan', linewidths=2, label='C' )
        plt.legend()
        plt.show()

    if len(points_category_A) == 0 and len(points_category_B) == 0 and len(points_category_C) == 0:
        print("No points near the corners!")
        return None

    point_tuples = {
                    'side':side,
                    'A': {'laser': cartesian_points[0], 'image': points_category_A },
                    'B': {'laser': cartesian_points[2], 'image': points_category_B },
                    'C': {'laser': cartesian_points[1], 'image': points_category_C },
                    }

    return point_tuples

if __name__ == "__main__":

    verbose = True

    csv_file_path = './calibration_data/output.csv'
    image_folder = './calibration_data/images'
    template_file = './calibration_data/board.jpg'

    range_image_publisher = RangeImagePublisher(csv_file_path, image_folder)

    # Send the model to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #PREPARE THE DETECTOR
    # Load the fine-tuned model's state dictionary
    root = "./calibration_data"
    save_path = os.path.join(root, 'one_shot_object_detector.pth')
    model_state_dict = torch.load(save_path, map_location=device)

    # Create an instance of the model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 1
    model.roi_heads.box_predictor.cls_score.out_features = num_classes
    model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
    model.load_state_dict(model_state_dict)

    model.to(device)
    model.eval()

    cmd = input("Press Enter to publish the next scan message...    [press Q to stop]")

    good_points = {
        'back':[],
        "left":[],
        "right":[],
        "front":[]
    }

    while cmd.lower() != 'q':
        (scan, image), progress = range_image_publisher.publish_next_scan_image()

        if scan is None:
             break

        print("+++++++++++++++++ Processing image {}/{}".format(progress[0], progress[1]))

#        if cmd.lower() == 's':
#            cmd = input("Press Enter to publish the next scan message...    [press Q to stop]")
#            continue

        laser_spec = range_image_publisher.get_laser_specs()
        laser_point = process2(scan, image, laser_spec)

        # #for testing
        # laser_point = [
        #     [ 2.30571884, -1.59577398],
        #     [ 2.24146292, -1.34847935],
        #     [ 2.03093913, -1.58731606] ]

        if laser_point is not None:
            point_tuples = process_image(np.array(laser_point), image, detector=model)
            print(point_tuples)

            if point_tuples is not None:

                side=point_tuples['side']

                if side is not None:

                    if point_tuples['A']['image'].shape[0] >=2 :
                        good_points[side[1]].append( [point_tuples['A']['laser'], point_tuples['A']['image']] )

                    if point_tuples['B']['image'].shape[0] >=2 :
                        good_points[side[1]].append( [point_tuples['B']['laser'], point_tuples['B']['image']] )

                    if point_tuples['C']['image'].shape[0] >=2 :
                        good_points[side[1]].append( [point_tuples['C']['laser'], point_tuples['C']['image']] )

        #cmd = input("Press Enter to publish the next scan message...    [press Q to stop]")


# Specify the file path where you want to save the dictionary
file_path = "cameraLaser_pointsUHD.pkl"

# save dictionary to person_data.pkl file
with open(file_path, 'wb') as fp:
    pickle.dump(good_points, fp)
    print('dictionary saved successfully to file')


print(f"Point for laser and camera calibration saved at {file_path}")