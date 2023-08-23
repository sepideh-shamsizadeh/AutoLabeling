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
from tqdm import tqdm


def main():

    #PERFORM INFERENCE
    #Detect board with the trained model
    folder_path =  "backgrounds"
    out_path = os.path.join(folder_path, "sides")
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    for i, image_path in tqdm(enumerate(image_paths)):

        print("Processing image: ", image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cube = CubeProjection(Image.fromarray(image), out_path)
        cube.cube_projection(face_id=None, img_id="split_img{}".format(i))


if __name__ == "__main__":

    main()