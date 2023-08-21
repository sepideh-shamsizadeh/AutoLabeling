import math
import cv2


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


def check_intersection(d_bound, point):
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
            if check_intersection(person, (u, v)):
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
                if check_intersection(person, (u, v)):
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
    for xy in XY_people:
        x, y = check_xy(xy, face)
        # u, v = convert_robotF2imageF(x, y, side_info)
        #         draw_circle_bndBOX(u, v, side_img)
        print((x, y))

    return XY_people


def draw_circle_bndBOX(u, v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)
