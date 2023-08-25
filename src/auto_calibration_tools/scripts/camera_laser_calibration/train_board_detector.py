import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
from tqdm import tqdm


# Define synthetic data generation function
class SyntheticDataset(Dataset):
    def __init__(self, target_image, target_image2, backgrounds, num_samples, device, augmentation_probability=1.0, use_advanced_aug=True):
        self.target_image = target_image
        self.target_image2 = target_image2.resize(target_image.size)
        self.backgrounds = backgrounds
        self.num_samples = num_samples
        self.adv_aug = use_advanced_aug
        self.augmentation_probability = augmentation_probability
        self.device = device

        # Get the dimensions of both images
        width1, height1 = self.target_image.size
        width2, height2 = self.target_image2.size
        separation_width = 60

        # Calculate the total width of the concatenated image
        new_width = width1 + width2 + separation_width  # Adjust separation_width as needed

        # Create a new blank image with the calculated dimensions
        concatenated_image = Image.new('RGBA', (new_width, max(height1, height2)))

        # Paste the first image on the left
        concatenated_image.paste(self.target_image2, (0, 0))  # Adjust separation_width as needed

        # Paste the second image to the right of the first image
        concatenated_image.paste(self.target_image, (width2 + separation_width, 0))

        # Update self.target_image
        self.target_image = concatenated_image

        random.seed(11)

    def __len__(self):
        return self.num_samples

    def apply_width_perspective_distortion2(self, image):
        width, height = image.size

        # Define random width distortion factors
        left_factor = random.uniform(0.1, 0.3)
        right_factor = random.uniform(0.7, 0.9)

        # Calculate the new left and right coordinates based on the width distortion factors
        left = left_factor * width
        right = right_factor * width

        # Define the perspective distortion matrix
        matrix = [
            left, 0,
            right, 0,
            width, height,
            0, height
        ]

        # Apply perspective transformation
        perspective = image.transform(image.size,  Image.BICUBIC, matrix)

        plt.subplots(2,1, 1)
        plt.imshow(np.array(image))
        plt.subplots(2, 1, 2)
        plt.imshow(np.array(perspective))
        plt.show()

        return perspective

    def apply_perspective_distortion(self, image):
        width, height = image.size
        # Define random perspective distortion points
        left = random.uniform(0.0, 0.49) * width
        right = random.uniform(0.51, 1.0) * width
        top = random.uniform(0.0, 0.49) * height
        bottom = random.uniform(0.51, 1.0) * height

        # Define the perspective distortion matrix
        matrix = [
            left, top,
            right, top,
            width, height,
            0, bottom
        ]

        # Apply perspective transformation
        perspective = image.transform(image.size, Image.BICUBIC, matrix)
        return perspective

    def __getitem__(self, idx):

        background_path = "./backgrounds/sides"
        while background_path == "./backgrounds/sides":
            background_path = random.choice(self.backgrounds)

        background = Image.open(background_path)

        # Randomly scale the target image
        if random.random() < self.augmentation_probability:
            scale_x = random.uniform(0.2, 1.25)
            scale_y = random.uniform(0.2, 1.25)
        else:
            scale_x = scale_y = 1.0

        target_resized = self.target_image.resize(
            (int(self.target_image.width * scale_x), int(self.target_image.height * scale_y)))

        # Randomly rotate the target image
        if random.random() < 0 and self.adv_aug:
            angle = random.randint(-90, 90)
            target_rotated = target_resized.rotate(angle, resample=Image.BICUBIC, expand=True)
        else:
            target_rotated = target_resized

        # Apply random perspective distortion to the rotated target image
        if random.random() < self.augmentation_probability and self.adv_aug:
            target_distorted = self.apply_perspective_distortion(target_rotated)
        else:
            target_distorted = target_rotated

        # Paste the distorted target image onto the background
        paste_x = random.randint(0, background.width - target_distorted.width)
        paste_y = random.randint(0, background.height - target_distorted.height)
        background.paste(target_distorted, (paste_x, paste_y), target_distorted)

        # Define bounding box coordinates (you need to adjust this based on your target image)
        bbox = [paste_x, paste_y, paste_x + target_distorted.width, paste_y + target_distorted.height]

        # Assuming bbox is in the format [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox

        # Calculate the width and height of the original bbox
        width = x_max - x_min
        height = y_max - y_min

        # Calculate the new half-width
        new_half_width = width / 2

        # Create two new bounding boxes
        bbox1 = [x_min, y_min, x_min + new_half_width, y_max]
        bbox2 = [x_min + new_half_width, y_min, x_max, y_max]

        # Transform the data into PyTorch tensors
        image_tensor = F.to_tensor(background).to(self.device)
        bbox_tensor1 = torch.tensor(bbox1, dtype=torch.float32).unsqueeze(0).to(self.device)
        bbox_tensor2 = torch.tensor(bbox2, dtype=torch.float32).unsqueeze(0).to(self.device)

        bbox_tensor = torch.cat([bbox_tensor1,bbox_tensor2], dim=0)
        #bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0).to(self.device)

        return {'image': image_tensor, 'bbox': {'boxes': bbox_tensor, 'labels': torch.tensor([1,1]).to(self.device)}}


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        torch.cuda.set_device(0)  # Specify the GPU device number if you have multiple GPUs
    else:
        device = torch.device("cpu")  # Use the CPU

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train a one-shot detector")

    # Add a command-line argument
    parser.add_argument('--folder', type=str, help='Path to the data folder', default=".")
    parser.add_argument('--adv_aug', dest='adv_aug', action='store_true')
    parser.set_defaults(adv_aug=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define paths
    root = args.folder
    backgrounds_folder = os.path.join(root,'backgrounds_UHD/sides')
    target_image_path = os.path.join(root,'target_UHD.png')  # The image with transparency
    target2_image_path = os.path.join(root,'target2_UHD.png')  # The image with transparency
    output_folder = 'synthetic_data/'
    test_image_folder = os.path.join(root, 'images_UHD/')

    # Load the target image (with transparency)
    target_image = Image.open(target_image_path).convert('RGBA')
    target_image2 = Image.open(target2_image_path).convert('RGBA')

    # Load a list of background images
    backgrounds = [os.path.join(backgrounds_folder, bg) for bg in os.listdir(backgrounds_folder)]

    # Create the synthetic dataset and DataLoader
    num_synthetic_samples = 500
    synthetic_dataset = SyntheticDataset(target_image, target_image2, backgrounds, num_synthetic_samples, device=device, use_advanced_aug=args.adv_aug)
    train_loader = DataLoader(synthetic_dataset, batch_size=2, shuffle=True)

    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    num_classes = 1
    model.roi_heads.box_predictor.cls_score.out_features = num_classes
    model.roi_heads.box_predictor.bbox_pred.out_features = num_classes * 4

    # Define an optimizer and a learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        total_loss = 0.0
        model.train()
        model.to(device)

        for batch in tqdm(train_loader):
            images = list(image for image in batch['image'])
            targets = []
            for i in range(len(images)):
                d = {}
                d['boxes'] = batch['bbox']['boxes'][i]
                d['labels'] = batch['bbox']['labels'][i]
                targets.append(d)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            # if batch_idx % 25 == 0:
            #     print(f"Batch [{batch_idx}/{len(train_loader)}] Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

        lr_scheduler.step()

    print("Training completed!")


    # Save the trained model
    save_path = os.path.join(root, 'one_shot_object_detector_5x3_UHD_SIDES.pth')
    torch.save(model.state_dict(), save_path)


    # Define a function for testing on images
    def visualize_predictions(model, test_image_folder):

        model.eval().cuda()

        with torch.no_grad():
            for image_name in os.listdir(test_image_folder):
                image_path = os.path.join(test_image_folder, image_name)
                test_image = Image.open(image_path).convert('RGB')
                test_tensor = F.to_tensor(test_image).unsqueeze(0).cuda()

                # Perform inference
                predictions = model(test_tensor)[0]

                # Visualize the image with predicted bounding boxes
                plt.imshow(test_image)
                ax = plt.gca()


                for box, score in zip(predictions['boxes'], predictions['scores']):
                    if score > 0.5:  # You can adjust this threshold
                        box = [float(coord) for coord in box]
                        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                                   color="orange", fill=False, linewidth=2))

                plt.axis('off')
                plt.show()

    # Test the trained model on images from a test folder
    visualize_predictions(model, test_image_folder)

    print("Visualization completed!")



if __name__ == "__main__":
    main()