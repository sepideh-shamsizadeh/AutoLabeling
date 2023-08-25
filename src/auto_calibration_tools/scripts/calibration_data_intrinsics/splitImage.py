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

def map_bounding_box_to_side(bounding_box, scale=1):
    u1, y1, u2, v2 = bounding_box

    if (u1 >= 0 and u2 < scale*240) or (u1 >= scale*1720 and u2 < scale*1960):
        if scale*240 <= v2 < scale*720:
            return (0, "back")
    elif u1 >= scale*240 and u2 < scale*720:
        if scale*240 <= v2 < scale*720:
            return (1, "left")
    elif u1 >= scale*720 and u2 < scale*1200:
        if scale*240 <= v2 <scale*720:
            return (2, "front")
    elif u1 >= scale*1200 and u2 < scale*1720:
        if scale*240 <= v2 < scale*720:
            return (3, "right")
    return None  # Bounding box doesn't fit any side


def main():

    #PREPARE THE DETECTOR
    # Load the fine-tuned model's state dictionary
    #root = "calibration_data_intrinsics"

    sides = ["back", "left", "right", "front","top", "bottom"]
    #create calibration folders
    for side in sides:
        if not os.path.exists(side):
            os.makedirs(side)
            print(f"Folder '{side}' created.")
        else:
            print(f"Folder '{side}' already exists.")

    model_state_dict = torch.load("one_shot_object_detectorUHD.pth")

    # Create an instance of the model
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 1
    model.roi_heads.box_predictor.cls_score.out_features = num_classes
    model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4
    model.load_state_dict(model_state_dict)
    
    # Send the model to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    #PERFORM INFERENCE
    #Detect board with the trained model
    folder_path =  "images_UHD2"
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    for i,image_path in tqdm(enumerate(image_paths)):

        print("Processing image: ", image_path) 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        test_tensor = F.to_tensor(image).unsqueeze(0).to(device)
        predictions = model(test_tensor)[0]

        if predictions['boxes'].shape[0] <= 0:
            print(f"!!! No detections in {image_path}")
            continue

        # Find the index of the bounding box with the highest score
        max_score_idx = torch.argmax(predictions['scores'])

        # Get the bounding box and score with the highest score
        max_score_box = predictions['boxes'][max_score_idx].detach().cpu().numpy()
        max_score = predictions['scores'][max_score_idx]

         # Enlarge the bounding box by a factor of 1.2 (adjust as needed)
        # enlargement_factor = 1.0
        # box = [
        #     int(np.round(
        #         max_score_box[0] - (max_score_box[2] - max_score_box[0]) * (enlargement_factor - 1) / 2)),
        #     int(np.round(
        #         max_score_box[1] - (max_score_box[3] - max_score_box[1]) * (enlargement_factor - 1) / 2)),
        #     int(np.round(
        #         max_score_box[2] + (max_score_box[2] - max_score_box[0]) * (enlargement_factor - 1) / 2)),
        #     int(np.round(
        #         max_score_box[3] + (max_score_box[3] - max_score_box[1]) * (enlargement_factor - 1) / 2)),
        # ]
        #
        # # Crop a slice of the image using the horizontal pixel coordinates
        # cropped_image = image[box[1]:box[3], box[0]:box[2]]

        # plt.imshow(cropped_image)
        # plt.show()

        side = map_bounding_box_to_side(max_score_box, scale = 2)

        if side is not None:
            cube = CubeProjection(Image.fromarray(image), ".")
            cube.cube_projection(face_id=side[0], img_id="int_cal_img{}".format(i))
            print(f"Detected board in {side} of {image_path}")
        else:
            print("Side is not well defined, skipping...")
        

if __name__ == "__main__":

    main()