import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse


# Define synthetic data generation function
class SyntheticDataset(Dataset):
    def __init__(self, target_image, backgrounds, num_samples, device, augmentation_probability=0.5, use_advanced_aug=True):
        self.target_image = target_image
        self.backgrounds = backgrounds
        self.num_samples = num_samples
        self.adv_aug = use_advanced_aug
        self.augmentation_probability = augmentation_probability
        self.device = device

        random.seed(11)

    def __len__(self):
        return self.num_samples

    def apply_perspective_distortion(self, image):
        width, height = image.size
        # Define random perspective distortion points
        left = random.uniform(0.1, 0.3) * width
        right = random.uniform(0.7, 0.9) * width
        top = random.uniform(0.1, 0.3) * height
        bottom = random.uniform(0.7, 0.9) * height

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
        background_path = random.choice(self.backgrounds)
        background = Image.open(background_path)

        # Randomly scale the target image
        if random.random() < self.augmentation_probability:
            scale_x = random.uniform(0.5, 1.5)
            scale_y = random.uniform(0.5, 1.5)
        else:
            scale_x = scale_y = 1.0

        target_resized = self.target_image.resize(
            (int(self.target_image.width * scale_x), int(self.target_image.height * scale_y)))

        # Randomly rotate the target image
        if random.random() < self.augmentation_probability and self.adv_aug:
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

        # Transform the data into PyTorch tensors
        image_tensor = F.to_tensor(background).to(self.device)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0).to(self.device)

        return {'image': image_tensor, 'bbox': {'boxes': bbox_tensor, 'labels': torch.tensor([1]).to(self.device)}}


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use the GPU
        torch.cuda.set_device(0)  # Specify the GPU device number if you have multiple GPUs
    else:
        device = torch.device("cpu")  # Use the CPU

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Train a one-shot detector")

    # Add a command-line argument
    parser.add_argument('--folder', type=str, help='Path to the data folder', default="calibration_data")
    parser.add_argument('--adv_aug', dest='adv_aug', action='store_true')
    parser.set_defaults(adv_aug=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define paths
    root = args.folder
    backgrounds_folder = os.path.join(root, 'backgrounds/')
    target_image_path = os.path.join(root,'target_image.png')  # The image with transparency
    output_folder = 'synthetic_data/'
    test_image_folder = os.path.join(root, 'images/')

    # Load the target image (with transparency)
    target_image = Image.open(target_image_path).convert('RGBA')

    # Load a list of background images
    backgrounds = [os.path.join(backgrounds_folder, bg) for bg in os.listdir(backgrounds_folder)]

    # Create the synthetic dataset and DataLoader
    num_synthetic_samples = 500
    synthetic_dataset = SyntheticDataset(target_image, backgrounds, num_synthetic_samples, device=device, use_advanced_aug=args.adv_aug)
    train_loader = DataLoader(synthetic_dataset, batch_size=4, shuffle=True)

    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
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

        for batch_idx, batch in enumerate(train_loader, 1):
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

            if batch_idx % 25 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}] Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Avg Loss: {avg_loss:.4f}")

        lr_scheduler.step()

    print("Training completed!")


    # Save the trained model
    save_path = os.path.join(root, 'one_shot_object_detector.pth')
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
                                                   color='red', fill=False, linewidth=2))

                plt.axis('off')
                plt.show()

    # Test the trained model on images from a test folder
    visualize_predictions(model, test_image_folder)

    print("Visualization completed!")



if __name__ == "__main__":
    main()