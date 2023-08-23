import math

import numpy as np
from scipy.optimize import minimize, least_squares
import pickle

def convert_robotF2imageF(tmpx, tmpy, side_info):

    H, fu, fv, v0, u0 = side_info

    Zc = H[4] * tmpx + H[5] * tmpy + H[8]
    u = ((fu * H[0] + u0 * H[4]) * tmpx + (fu * H[1] + u0 * H[5]) * tmpy + fu * H[6] + u0 * H[8]) / Zc
    v = ((fv * H[2] + v0 * H[4]) * tmpx + (fv * H[3] + v0 * H[5]) * tmpy + fv * H[7] + v0 * H[8]) / Zc
    return [u, v]

def compute_error1(H, A, B, X, Y, fu, fv, v0, u0):
    X = np.array(X)
    Y = np.array(Y)
    u = ((fu * H[0] + u0 * H[4]) * X + (fu * H[1] + u0 * H[5]) * Y + fu * H[6] + u0 * H[8]) / (H[4] * X + H[5] * Y + H[8])
    v = ((fv * H[2] + v0 * H[4]) * X + (fv * H[3] + v0 * H[5]) * Y + fv * H[7] + v0 * H[8]) / (H[4] * X + H[5] * Y + H[8])
    error = np.sum((A * u + B * v - 1)**2)
    return error


def compute_error2(H, A, B, X, Y, fu, fv, v0, u0):
    X = np.array(X)
    Y = np.array(Y)
    u = ((fu * H[0] + u0 * H[4]) * X + (fu * H[1] + u0 * H[5]) * Y + fu * H[6] + u0 * H[8]) / (
                H[4] * X + H[5] * Y + H[8])
    v = ((fv * H[2] + v0 * H[4]) * X + (fv * H[3] + v0 * H[5]) * Y + fv * H[7] + v0 * H[8]) / (
                H[4] * X + H[5] * Y + H[8])

    A = np.array(A)  # Convert A to a NumPy array
    B = np.array(B)  # Convert B to a NumPy array

    error = np.sum(np.abs(A * u + B * v - 1)) / math.sqrt(
        np.sum(A ** 2 + B ** 2))  # Use np.abs() for element-wise absolute value
    return error

def compute_H(parameters, fu, fv, u0, v0):

    A, B, X, Y = parameters[:,0], parameters[:,1], parameters[:,2], parameters[:,3]

    k = np.zeros((9, 9))
    for i in range(9):
        k[i, :] = [A[i] * fu * X[i], A[i] * fu * Y[i], B[i] * fv * X[i], B[i] * fv * Y[i],
                   (A[i] * u0 + B[i] * v0 - 1) * X[i], (A[i] * u0 + B[i] * v0 - 1) * Y[i],
                   A[i] * fu, B[i] * fv, (A[i] * u0 + B[i] * v0 - 1)]

    r = 0.000001 + 0.000005 * np.random.rand(9, 1)
    #print(r)

    k_inverse = np.linalg.pinv(k)
    H = np.dot(k_inverse, r)

    H_matrix = H.reshape(3, 3)
    return H_matrix


def optimizeH(H0, parameters, fu, fv, v0, u0):

    A, B, X, Y = parameters[:, 0], parameters[:, 1], parameters[:, 2], parameters[:, 3]
    optimization_function = lambda H: compute_error1(H, A, B, X, Y, fu, fv, v0, u0)
    H = minimize(optimization_function, H0).x
    return H

def Hoptimizationlsqnonlin(H0, parameters, fu, fv, v0, u0):

    A, B, X, Y = parameters[:, 0], parameters[:, 1], parameters[:, 2], parameters[:, 3]
    optimization_function = lambda H: compute_error2(H, A, B, X, Y, fu, fv, v0, u0)
    options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxfev': 1000}
    H = least_squares(optimization_function, H0, method='trf', xtol=options['xtol'], ftol=options['ftol'], max_nfev=options['maxfev']).x
    return H

def calculate_coefficients(u0, v0, u1, v1):

    #print((u1*v0-v1*u0))
    B = (u1 - u0)/((u1*v0-v1*u0)+0.0000001)
    #print(B)
    A= (1-B*v0)/(u0+0.00001)

    return A, B



if __name__ == '__main__':

    ####### CAMERA MATRIX (TRIAL)
    calib_file = "../calibration_data_intrinsics/intrinsics.pkl"
    # Open the pickle file for reading in binary mode
    with open(calib_file, 'rb') as file:
        # Load the dictionary from the pickle file
        camera_calib = pickle.load(file)

    # Specify the path of the calibration data
    file_path = "cameraLaser_points.pkl"

    # Open the file in binary read mode
    with open(file_path, 'rb') as file:
        # Load the dictionary from the pickle file
        data = pickle.load(file)

    for side, points in data.items():

        # points = list of tuples like ( (C, left),(B, right) )
        # left/right numpy array 5x2

        print("*****Processing side {}".format(side))
        print("*****Found {} tuples".format(len(points)))

        parameters = []

        for point in points:

            for tuple in point:

                laser_point, board_points = tuple

                u_coors = board_points[:, 0]  # Extracts the first column (u coordinates)
                v_coords = board_points[:, 1]  # Extracts the second column (v coordinates)
                X = laser_point[0]
                Y = laser_point[1]

                A1, B1 = calculate_coefficients( u_coors[0], v_coords[0],  u_coors[-1], v_coords[-1])
                param = (A1, B1, X, Y)
                parameters.append(param)

                A2, B2 = calculate_coefficients( u_coors[1], v_coords[1],  u_coors[-2], v_coords[-2])
                param = (A2, B2, X, Y)
                parameters.append(param)


        #Start Optimization
        parameters = np.array(parameters)
        k = camera_calib[side]['K']

        fu = k[0,0]
        fv = k[1,1]
        u0 = k[0,2]
        v0 = k[1,2]

        if parameters.shape[0] < 9:
            print("You have not enough data for side {}, skipping".format(side))
            continue


        H = compute_H(parameters, fu, fv, u0, v0)
        #print(H)
        H1 = optimizeH(H, parameters, fu, fv, u0, v0)
        #print(H1)

        H2 = Hoptimizationlsqnonlin(H1, parameters, fu, fv, u0, v0)
        print(H2)

        side_info = (H2, fu, fv, v0, u0)

        avg_error = 0
        cont = 0
        for point in points:

            for tuple in point:
                laser_point, board_points = tuple

                U,V = convert_robotF2imageF(laser_point[0], laser_point[1], side_info)

                #print("**** U {} V {}".format(u,v))
                for j, point in enumerate(board_points, 1):
                    u,v = point
                    #print("    u {} v {}".format(u,v))

                    avg_error += (U-u)**2
                    cont += 1

        avg_error = np.sqrt(avg_error) / cont
        print("AVG ERROR: {}".format(avg_error))


        print("************************************")