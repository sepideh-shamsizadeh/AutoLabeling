# !/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import argparse
import os
import matplotlib.pyplot as plt
import pickle

# Create an argument parser
parser = argparse.ArgumentParser(description="Train a one-shot detector")

# Add a command-line argument
parser.add_argument('--size_w', type=int, help='Widht of the board in corners', default=6)
parser.add_argument('--size_h', type=int, help='Height of the board in corners', default=9)
parser.add_argument('--square_size', type=float, help='Size of a square in the chessboard', default=0.025)

# Parse the command-line arguments
args = parser.parse_args()

# Defining the dimensions of checkerboard
CHECKERBOARD = (args.size_h, args.size_w)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
square_size = args.square_size  # Set the size of your squares here (in meters)

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = square_size * np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

sides = ["back", "left", "right", "front"]
results = {}

for side in sides:
    folder_path = f"./{side}"  # Replace with the actual path to your dataset

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder '{side}' does not exist.")
        continue

    print("*** CALIBRATION OF: ", side)

    # Get a list of image file names in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".jpg", ".jpeg", ".png", ".tif"))]

    # Iterate through the image files
    for image_file in image_files:

        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        cv2.imshow('img', img)
        # cv2.imwrite(fname, img)
        cv2.waitKey(300)

    cv2.destroyAllWindows()

    h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("Camera Matrix \n:", mtx)
    print("total error: {}".format(mean_error / len(objpoints)))

    results[side] = {"K":mtx, "dist":dist, "rvecs":rvecs, "tvecs":tvecs, "rpj_error": mean_error / len(objpoints) }

# Specify the file path where you want to save the dictionary
file_path = "intrinsics.pkl"

# save dictionary to person_data.pkl file
with open(file_path, 'wb') as fp:
    pickle.dump(results, fp)
    print('Dictionary saved successfully to file')

print(f"Intrinsics calibration results saved to {file_path}")


