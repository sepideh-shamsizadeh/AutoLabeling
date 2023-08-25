from PIL import Image
from math import pi, sin, cos, tan, atan2, hypot, floor
from numpy import clip
import glob
import os
import numpy as np


class CubeProjection:
    def __init__(self, imgIn, output_path):
        self.output_path = output_path
        self.imagin = imgIn
        self.sides = []

    def cube_projection(self, face_id=None, img_id="img"):
        imgIn = self.imagin
        inSize = imgIn.size
        faceSize = int(inSize[0]/4)
        FACE_NAMES = {
            0: 'back',
            1: 'left',
            2: 'front',
            3: 'right',
            4: 'top',
            5: 'bottom'
        }

        if face_id is None:
            for face in range(6):
                imgOut = Image.new('RGB', (faceSize, faceSize), 'black')
                side = self.convertFace(imgIn, imgOut, face)
                if self.output_path != '':
                    save_path = os.path.join(self.output_path,FACE_NAMES[face],img_id+'.jpg')
                    print("SAVED: ", save_path)
                    imgOut.save(save_path)
                else:
                    self.sides.append({FACE_NAMES[face]: side})
        else:
            face = face_id
            imgOut = Image.new('RGB', (faceSize, faceSize), 'black')
            side = self.convertFace(imgIn, imgOut, face)
            if self.output_path != '':
                save_path = os.path.join(self.output_path,FACE_NAMES[face],img_id+'.jpg')
                print("SAVED: ", save_path)
                imgOut.save(save_path)
            else:
                self.sides.append({FACE_NAMES[face]: side})


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

    def convertFace(self,imgin, imgout,faceIdx):
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

    def outImg2XYZ2(self, i, j, faceIdx, faceSize):
        a = 2.0 * i / faceSize
        b = 2.0 * j / faceSize

        if faceIdx == 0:  # back
            return (-1.0, 1.0 - a, 1.0 - b)
        elif faceIdx == 1:  # left
            return (a - 1.0, -1.0, 1.0 - b)
        elif faceIdx == 2:  # front
            return (1.0, a - 1.0, 1.0 - b)
        elif faceIdx == 3:  # right
            return (1.0 - a, 1.0, 1.0 - b)
        elif faceIdx == 4:  # top
            return (b - 1.0, a - 1.0, 1.0)
        elif faceIdx == 5:  # bottom
            return (1.0 - b, a - 1.0, -1.0)

    def convertFace2(self, imgin, imgout, faceIdx):
        inSize = np.array(imgin.size)
        outSize = np.array(imgout.size)
        inpix = np.array(imgin)
        outpix = np.zeros_like(inpix)
        facesize = outSize[0]

        xout, yout = np.meshgrid(range(facesize), range(facesize))
        xout, yout = xout.flatten(), yout.flatten()

        (x, y, z) = self.outImg2XYZ2(xout, yout, faceIdx, facesize)
        theta = np.arctan2(y, x)
        r = np.hypot(x, y)
        phi = np.arctan2(z, r)

        # Source img coords
        uf = 0.5 * inSize[0] * (theta + np.pi) / np.pi
        vf = 0.5 * inSize[0] * (np.pi / 2 - phi) / np.pi

        # Use bilinear interpolation between the four surrounding pixels
        ui = np.floor(uf).astype(int)  # coord of pixel to bottom left
        vi = np.floor(vf).astype(int)
        u2 = ui + 1  # coords of pixel to top right
        v2 = vi + 1
        mu = uf - ui  # fraction of way across pixel
        nu = vf - vi

        # Pixel values of four corners
        A = inpix[ui % inSize[0], np.clip(vi, 0, inSize[1] - 1)]
        B = inpix[u2 % inSize[0], np.clip(vi, 0, inSize[1] - 1)]
        C = inpix[ui % inSize[0], np.clip(v2, 0, inSize[1] - 1)]
        D = inpix[u2 % inSize[0], np.clip(v2, 0, inSize[1] - 1)]

        # Interpolate
        r = A * (1 - mu) * (1 - nu) + B * mu * (1 - nu) + C * (1 - mu) * nu + D * mu * nu

        outpix[xout, yout] = r.reshape((facesize, facesize, 3)).astype(int)
        imgout.putdata([tuple(pixel) for pixel in outpix.reshape(-1, 3)])


if __name__ == '__main__':
    images = glob.glob('/home/sepid/workspace/Thesis/GuidingRobot/src/calib/checkerboard/*.png')
    for fname in images:
        print(fname)
        print(fname.split('/'))
        name = [z for z in fname.split('/') if '.png' in z][0]
        print(name.split('.')[0])
        imgIn = Image.open(fname)
        cube = CubeProjection(imgIn, '/home/sepid/workspace/Thesis/GuidingRobot/src/calib/checkerboard/'+name.split('.')[0])
        cube.cube_projection()