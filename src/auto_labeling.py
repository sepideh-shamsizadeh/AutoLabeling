import yaml
import csv
import cv2
import ast
import math
import os
from PIL import Image
from math import pi, atan2, hypot, floor
from numpy import clip
import numpy as np
import detect_people

model = detect_people.load_model()


def counter():
    num = 0
    while True:
        yield num
        num += 1


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
        offset = 30
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


# def check_points(x_y, p_uv, person):
#     avgx = (person[0] + person[2]) / 2
#     avgy = (person[1] + person[3]) / 2
#     # print('avg', avgx, avgy)
#     closest_distance = float('inf')
#     closest_point = None
#
#     for i, p in enumerate(p_uv):
#         u = p[0]
#         v = p[1]
#         print('1', u, v)
#         if not person[0] <= p[0] <= person[2]:
#             if p[0] <= person[0]:
#                 u = p[0] + 35
#             elif p[0] >= person[2]:
#                 u = p[0] - 35
#         if not person[1] <= p[1] <= person[3]:
#             if p[1] <= person[1]:
#                 v = p[1] + 35
#             elif p[1] >= person[3]:
#                 v = p[1] - 35
#         print('2', u, v)
#         dist = distance(u, v, avgx, avgy)
#         if dist < closest_distance:
#             closest_distance = dist
#             closest_point = x_y[i]
#     return closest_point

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


def selected_point(side_xy, side, side_info, face, detected):
    print(face)
    XY_people = []
    print(detected)
    for ind, person in enumerate(detected):
        flag = False
        p = []
        x_y = []
        for xy in side_xy:
            x, y = check_xy(xy, face)
            u, v = convert_robotF2imageF(x, y, side_info)
            if u < 0:
                u = 0
            if v < 0:
                v = 0
            # if face == 'back':
            #     v -= 90
            #     u += 30
            # if face == 'left':
            #     u += 30
            #     v -= 20
            # print('------------------------------')
            # print('u,v', u, v)
            # print('(u, v)', (u, v))
            # print('x,y', (xy[0], xy[1]))
            if check_intersection(person, (u, v), False):
                # print('+++++++++++++++++++++++++++++++++++++++++++++')
                # print(face)
                # print(person)
                # print('(u, v)', (u, v))
                # print('x,y', (xy[0], xy[1]))
                draw_circle_bndBOX(u, v, cv_image)
                flag = True
                p.append((u, v))
                x_y.append((xy[0], xy[1]))
                if face =='back':
                    filename = 'back' + str(next(counter_gen)) + '.jpg'
                    print(filename)
                    cv2.imwrite(filename, cv_image)

        x = 0
        y = 0
        # print('xy', x_y)
        if not flag:
            for xy in side:
                x, y = check_xy(xy, face)
                u, v = convert_robotF2imageF(x, y, side_info)
                if v > 480:
                    v = 479
                if face == 'left':
                    # print('******************************************************')
                    # print(person)
                    # print(xy)
                    # print(x, y)
                    # print(u, v)
                    u += 50
                    v -= 40
                    # draw_circle_bndBOX(u, v, cv_image)
                if check_intersection(person, (u, v), True):
                    # print(u, v)
                    # draw_circle_bndBOX(u, v, cv_image)
                    # print('//////////////////////////////////////////////////////////')
                    # print(x, y)
                    # print(u, v)
                    # print(person)
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
        if x != 0 and y != 0:
            XY_people.append((x, y))
    for xy in XY_people:
        x, y = check_xy(xy, face)
        u, v = convert_robotF2imageF(x, y, side_info)
        # print(x, y)
        # print(u, v)
        # print(person)
        draw_circle_bndBOX(u, v, cv_image)
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
    scan = []
    with open('/data1/scan.csv', 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        for row in reader:
            image_id = int(row[0])  # Extract the image ID from the first column
            ranges = [float(value) for value in row[1:]]  # Extract th
            scan.append(ranges)

    dr_spaam = []
    with open('/data1/drspaam_data2.csv', 'r') as file:
        # Create a CSV reader object
        reader = csv.reader(file)
        # Read each row of the CSV file
        for row in reader:
            dr_spaam.append(row)
    print(len(dr_spaam))
    data = {}
    # for i in range(36, int(len(scan)/2)):
    for i in range(14, 15):

        path = '/home/sepid/workspace/Thesis/GuidingRobot/data1/image_' + str(i) + '.jpg'
        print(path)
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
            # print(back_xy, left_xy, right_xy, front_xy)
            # print(left_xy)
            # cv2.imshow('img1', img)
            # cv2.waitKey(0)
            img = Image.fromarray(img)
            sides = CubeProjection(img, '')
            sides.cube_projection()
            people = []
            people_img = []
            print('drdrdrdrdrdrdrdrdrdr', dr_spaam[i])
            for face, side_img in sides.sides.items():
                if face in FACE_NAMES:
                    cv_image = np.array(side_img)
                    detected = detect_people.detect_person(cv_image, model)
                    print(detected)
                    if face == 'back':
                        XY = selected_point(back_xy, back, back_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'front':
                        XY = selected_point(front_xy, front, front_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'right':
                        XY = selected_point(right_xy, right, right_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

                    elif face == 'left':
                        XY = selected_point(left_xy, left, left_info, face, detected)
                        for xy in XY:
                            if xy[0] != 0 and xy[1] != 0:
                                people.append((xy[0], xy[1]))

            people = list(dict.fromkeys(people))
            print(people)
            # if len(people) > 0:
            #     fid = next(counter_gen)
            #     # file_name = '/home/sepid/workspace/Thesis/GuidingRobot/data2/output01.yaml'
            #     print(fid)
            #     # write_output(people, fid, file_name)
            #     output_file = '/home/sepid/workspace/Thesis/GuidingRobot/data2/output02.yaml'
            #     with open(output_file, 'a') as file:
            #         # Write the YAML data to the file
            #         yaml_data = {'frame ' + str(fid): path}
            #         yaml.dump(yaml_data, file)
            # # #
            # # file_name = '/home/sepid/workspace/Thesis/GuidingRobot/data0/outputi.yaml'
            # # write_output(people, i, file_name)
