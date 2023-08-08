import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import cv2

from read_calib_data import RangeImagePublisher


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

    plt.xlabel('X Distance (m)')
    plt.ylabel('Y Distance (m)')
    plt.title('Laser Scan Data')
    plt.axis('equal')  # Equal aspect ratio
    plt.grid(True)
    #plt.show()

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

    plt.scatter(A[0], A[1], c='cyan', marker='o', s=400, linewidths=3, label="Final prediction A", alpha=0.5)
    plt.scatter(B[0], B[1], c='red', marker='o', s=400, linewidths=3, label="Final prediction B", alpha=0.5)
    plt.scatter(C[0], C[1], c='blue', marker='o', s=400, linewidths=3, label="Final prediction C", alpha=0.5)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.legend()

    # Convert Cartesian to polar coordinates
    cartesian_points = np.array(final_point)
    polar_coordinates = cartesian_to_polar(cartesian_points)

    print(polar_coordinates)

    plt.show()

    if min_dist < 0.180:
        return final_point
    else:
        print("Confidence is too low! Discard the detection...")
        return None

# def cartesian_to_polar(cartesian_points):
#     x, y = cartesian_points[:, 0], cartesian_points[:, 1]
#     angles = np.arctan2(y, x)  # Calculate angles in radians
#     angles_deg = np.degrees(angles)  # Convert angles to degrees
#
#     # Map angles to the [-180°, +180°] range
#     angles_deg = (angles_deg + 180) % 360 - 180
#
#     # Calculate magnitudes (distances)
#     magnitudes = np.sqrt(x**2 + y**2)
#
#     return np.column_stack((angles_deg, magnitudes))

def cartesian_to_polar(cartesian_points):

    x, y = cartesian_points[:, 0], cartesian_points[:, 1]
    angles = np.arctan2(y, x)  # Calculate angles in radians
    angles_deg = np.degrees(angles)  # Convert angles to degrees

    # Map angles to the [0°, 360°] range
    angles_deg = (angles_deg + 360) % 360

    # Calculate magnitudes (distances)
    magnitudes = np.sqrt(x**2 + y**2)

    return np.column_stack((angles_deg, magnitudes))


# def map_polar_to_horizontal_pixel(polar_points, image_width):
#     angle_per_pixel = 360.0 / image_width
#
#     horizontal_pixel_coords = []
#     for angle_deg, _ in polar_points:
#         # Calculate horizontal pixel coordinate
#         pixel_coord = int((angle_deg + 180) / angle_per_pixel)
#         horizontal_pixel_coords.append(pixel_coord)
#
#     return horizontal_pixel_coords

def map_polar_to_horizontal_pixel(polar_points, image_width):
    angle_per_pixel = 360.0 / image_width

    horizontal_pixel_coords = []
    for angle_deg, _ in polar_points:
        # Calculate horizontal pixel coordinate
        pixel_coord = image_width - int(angle_deg / angle_per_pixel)
        horizontal_pixel_coords.append(pixel_coord)

    return horizontal_pixel_coords

def process_image(cartesian_points, image, template=None):

    # Convert Cartesian to polar coordinates
    polar_coordinates = cartesian_to_polar(cartesian_points)

    # Image width (number of pixels)
    image_heigth, image_width, ch = image.shape

    # Map polar coordinates to horizontal pixel coordinates
    horizontal_pixel_coordinates = map_polar_to_horizontal_pixel(polar_coordinates, image_width)

    print(horizontal_pixel_coordinates)

    end = max(horizontal_pixel_coordinates[1], horizontal_pixel_coordinates[2])
    start = min(horizontal_pixel_coordinates[1], horizontal_pixel_coordinates[2])

    rescale = (end-start)//2

    offset = 50

    start = max(start-rescale-offset, 0)
    end = min(end+rescale-offset, image_width-1)

    # Crop a slice of the image using the horizontal pixel coordinates
    cropped_image = image[500:,start:end]

    # Display the original and cropped images
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    # only one line may be specified; full height
    plt.axvline(x=start, color='blue')
    plt.axvline(x=end, color='red')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image)
    plt.title("Cropped Image")

    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # detect corners with the goodFeaturesToTrack function.
    corners = cv2.goodFeaturesToTrack(gray_image, 6, 0.05, 10)
    corners = np.int0(corners)

    # Mark detected corners with red dots
    plt.plot(corners[:,:, 1], corners[:,:, 0], 'r.', markersize=5)


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file_path = './calibration_data/output.csv'
    image_folder = './calibration_data/images'

    range_image_publisher = RangeImagePublisher(csv_file_path, image_folder)

    cmd = ''
    while cmd.lower() != 'q':
        scan, image = range_image_publisher.publish_next_scan_image()
        laser_spec = range_image_publisher.get_laser_specs()
        laser_point = process(scan, image, laser_spec)

        # laser_point = [
        #     [ 2.30571884, -1.59577398],
        #     [ 2.24146292, -1.34847935],
        #     [ 2.03093913, -1.58731606] ]

        if laser_point is not None:
            process_image(np.array(laser_point), image)

        #cmd = input("Press Enter to publish the next scan message...    [press Q to stop]")
