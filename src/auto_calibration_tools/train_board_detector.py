import os
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define paths
root = "./calibration_data"
backgrounds_folder = os.path.join(root, 'backgrounds/')
target_image_path = os.path.join(root,'target_image.png')  # The image with transparency
output_folder = 'synthetic_data/'
test_image_folder = os.path.join(root, 'images/')

# Load the target image (with transparency)
target_image = Image.open(target_image_path).convert('RGBA')

# Load a list of background images
backgrounds = [os.path.join(backgrounds_folder, bg) for bg in os.listdir(backgrounds_folder)]


# Define synthetic data generation function
class SyntheticDataset(Dataset):
    def __init__(self, target_image, backgrounds, num_samples):
        self.target_image = target_image
        self.backgrounds = backgrounds
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        background_path = random.choice(self.backgrounds)
        background = Image.open(background_path)

        # Randomly scale the target image
        scale_x = random.uniform(0.5, 1.5)
        scale_y = random.uniform(0.5, 1.5)

        target_resized = self.target_image.resize(
            (int(self.target_image.width * scale_x), int(self.target_image.height * scale_y)))

        # Paste the target image onto the background
        paste_x = random.randint(0, background.width - target_resized.width)
        paste_y = random.randint(0, background.height - target_resized.height)
        background.paste(target_resized, (paste_x, paste_y), target_resized)

        # Define bounding box coordinates (you need to adjust this based on your target image)
        bbox = [paste_x, paste_y, paste_x + target_resized.width, paste_y + target_resized.height]

        # Transform the data into PyTorch tensors
        image_tensor = F.to_tensor(background).cuda()
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0).cuda()

        return {'image': image_tensor, 'bbox': {'boxes':bbox_tensor, 'labels':torch.tensor([1]).cuda()}}


# Create the synthetic dataset and DataLoader
num_synthetic_samples = 100
synthetic_dataset = SyntheticDataset(target_image, backgrounds, num_synthetic_samples)
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
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    total_loss = 0.0
    model.train()
    model.cuda()

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

        if batch_idx % 10 == 0:
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