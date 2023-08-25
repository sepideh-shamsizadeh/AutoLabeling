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
import random


def map_bounding_box_to_side(bounding_box, scale=1):
    u1, y1, u2, v2 = bounding_box

    if (u1 >= 0 and u2 < scale * 240) or (u1 >= scale * 1720 and u2 < scale * 1960):
        if scale * 240 <= v2 < scale * 720:
            return (0, "back")
    elif u1 >= scale * 240 and u2 < scale * 720:
        if scale * 240 <= v2 < scale * 720:
            return (1, "left")
    elif u1 >= scale * 720 and u2 < scale * 1200:
        if scale * 240 <= v2 < scale * 720:
            return (2, "front")
    elif u1 >= scale * 1200 and u2 < scale * 1720:
        if scale * 240 <= v2 < scale * 720:
            return (3, "right")
    return None  # Bounding box doesn't fit any side


def main():
    # PERFORM INFERENCE
    # Detect board with the trained model
    folder_path = "backgrounds_UHD"
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                   filename.endswith(('.jpg', '.png', '.jpeg'))]

    for i, image_path in tqdm(enumerate(image_paths)):

        print("Processing image: ", image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cube = CubeProjection(Image.fromarray(image), ".")
        cube.cube_projection(face_id=None, img_id="backgrund{}".format(i))


if __name__ == "__main__":
    main()