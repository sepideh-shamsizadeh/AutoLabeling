import math
import cv2
import numpy as np


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




def draw_circle_bndBOX(u,v, img):
    cv2.circle(img, (int(u), int(v)), 10, (0, 0, 255), 3)
    cv2.imshow('image', img)
    cv2.waitKey(0)


f = '/home/sepid/workspace/Thesis/GuidingRobot/src/calib/ABback.txt'
A = []
B = []
xt = []
yt = []

back_info = {
    'H': np.array([ 3.72265219e-05,1.64017221e-04,-1.69142004e-05,-1.00768104e-05
,-1.70414355e-04,2.86273496e-05,-1.74768952e-06,-1.51941775e-04
,3.07150681e-05]),
    'fu': 250.001420127782,
    'fv': 253.955300723887,
    'u0': 239.731339559399,
    'v0': 246.917074981568
}

with open(f, 'r') as file:
    for line in file:
        line = line.strip()
        if line.endswith('.jpg'):
            filename = line
            print(filename)
            img = cv2.imread('/home/sepid/workspace/Thesis/GuidingRobot/src/auto_labeling/calib/scenes/back/'+filename)
        elif line.endswith(')'):
            A_value, B_value = line.strip('()\n').split(',')
            line = file.readline()
            xt_value, yt_value, _ = map(float, line.split(';'))
            u,v = convert_robotF2imageF(xt_value, yt_value, back_info)
            draw_circle_bndBOX(u,v,img)


