import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import math
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Set the random seed using the current time
random.seed(time.time())


# Generate training data
# For this example, we'll generate random points as positive and negative examples.
# You should replace this with your actual training data.
def generate_positive_samples(template, num_samples):
    # Apply random transformations to the template to generate positive samples
    positive_samples = []
    for _ in range(num_samples):

        while True:
            rotation_angle = np.random.uniform(0, 360)
            x_offset = np.random.uniform(-25, 25)
            y_offset = np.random.uniform(-25, 25)

            rotation_angle_rad = np.radians(rotation_angle)
            rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                                        [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]])

            new_sample = []
            for point in template:
                rotated_point = np.dot(rotation_matrix, np.array(point).T)
                translated_point = rotated_point + np.array([x_offset, y_offset])
                new_sample.append(translated_point)

            if np.linalg.norm(new_sample[1]) < np.linalg.norm(new_sample[0]) and np.linalg.norm(
                    new_sample[2]) < np.linalg.norm(new_sample[0]):
                break  # Constraint satisfied

        positive_samples.append(new_sample)

    return positive_samples


def generate_negative_samples(template, num_samples):

    negative_samples = []
    for _ in range(num_samples):

        template = []
        for j in range(3):
            x = np.random.uniform(-0.3, 0.3)
            y = np.random.uniform(-0.3, 0.3)
            template.append([x, y])

        # Generate random points as negative samples
        rotation_angle = np.random.uniform(0, 360)
        x_offset = np.random.uniform(-25, 25)
        y_offset = np.random.uniform(-25, 25)

        rotation_angle_rad = np.radians(rotation_angle)
        rotation_matrix = np.array([[np.cos(rotation_angle_rad), -np.sin(rotation_angle_rad)],
                                    [np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)]])

        new_sample = []
        for point in template:
            rotated_point = np.dot(rotation_matrix, np.array(point).T)
            translated_point = rotated_point + np.array([x_offset, y_offset])
            new_sample.append(translated_point)

        negative_samples.append(new_sample)

    return negative_samples

# Define a simple feedforward neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Two output classes (0 and 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
def main():
    gt_template = [
        [0, 0],
        [0.285, 0],
        [0, 0.285]
    ]

    # Combine positive and negative samples
    positive_samples = generate_positive_samples(gt_template, 10000)
    negative_samples = generate_negative_samples(gt_template, 10000)

    ps = np.array((positive_samples))
    ns = np.array((negative_samples))

    # plt.scatter(ps[:,:,0], ps[:,:,1], marker='x', color='blue')
    # plt.scatter(ns[:,:,0], ps[:,:,1], marker='x', color='red')
    #
    # plt.axis('equal')  # Equal aspect ratio
    # plt.show()


    X = positive_samples + negative_samples
    y = [1] * len(positive_samples) + [0] * len(negative_samples)

    # Convert data to numpy arrays
    X = np.array(X).reshape(-1, 6)  # Each sample has 2 points (6 values)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)  # Assuming you have two classes (0 and 1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the neural network
    input_size = 6
    model = NeuralNetwork(input_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        avg_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            avg_loss += loss
            loss.backward()
            optimizer.step()

        print("Loss: ", avg_loss / len(train_loader))

    # Evaluate the classifier
    X_test = torch.Tensor(X_test)
    y_pred = model(X_test).argmax(dim=1).numpy()

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)

    # Save the trained PyTorch model
    torch.save(model.state_dict(), 'laserpoint_detector.pth')

if __name__ == "__main__":
    main()


