import os
import time
import random
import zipfile
from itertools import chain

import timm
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from LATransformer.model import ClassBlock, LATransformer, LATransformerTest
from LATransformer.utils import save_network, update_summary, get_id
from LATransformer.metrics import rank1, rank5, rank10, calc_map

import faiss

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This will make sure no GPU is being used
device = "cpu"
batch_size = 8
gamma = 0.7
seed = 42

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base = vit_base.to(device)

# Create La-Transformer
model = LATransformerTest(vit_base, lmbd=8).to(device)

# Load LA-Transformer
name = "la_with_lmbd_8"
save_path = os.path.join('Weights-20230803T150538Z-001/Weights/net_best.pth')
model.load_state_dict(torch.load(save_path), strict=False)
print(model.eval())
transform_query_list = [
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform_gallery_list = [
    transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
data_transforms = {
    'query': transforms.Compose(transform_query_list),
    'gallery': transforms.Compose(transform_gallery_list),
}

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def extract_feature_single(model, image):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
    return output.squeeze().detach().cpu()

def load_and_preprocess_image(image_path):
    transform_list = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_and_preprocess_images(image1_path, image2_path):
    image1 = load_and_preprocess_image(image1_path)
    image2 = load_and_preprocess_image(image2_path)
    return image1, image2

def calculate_similarity(query_vector, gallery_vectors):
    # Normalize the query vector
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Normalize the gallery vectors
    gallery_vectors = gallery_vectors / np.linalg.norm(gallery_vectors)

    # Perform cosine similarity between query and gallery vectors
    similarity_scores = np.dot(query_vector, gallery_vectors.T)
    return similarity_scores

# Load and preprocess the gallery images
gallery_image1_path = "data/2.jpg"
query_image_path = "data/3.jpg"
gallery_image1 = load_and_preprocess_image(gallery_image1_path)

# Extract features from the gallery images
gallery_vector1 = extract_feature_single(model, gallery_image1)

# Prepare the gallery vectors for Faiss search
gallery_vector1 = gallery_vector1.view((-1)).numpy()

# Combine the gallery vectors into a single array for Faiss


# Load and preprocess the query image
query_image = load_and_preprocess_image(query_image_path)

# Extract features from the query image
query_vector = extract_feature_single(model, query_image)

# Prepare the query vector for Faiss search
query_vector = query_vector.view((-1)).numpy()

# Calculate similarity scores between the query vector and gallery vectors
similarity_scores_1 = calculate_similarity(query_vector, gallery_vector1)

# Print the similarity scores
print("Similarity Score for Query and Gallery Image 1:")
print(similarity_scores_1)




