import faiss
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def get_activation(name, activation):
    def hook(model1, input, output):
        activation[name] = output.detach()

    return hook


def extract_feature_single(model1, image, device="cpu"):
    model1.eval()
    image = image.to(device)
    with torch.no_grad():
        output = model1(image)
    return output[0].squeeze().detach().cpu()


def load_and_preprocess_image(image, bounding_box):
    # Validate if the bounding_box is a list with 4 elements (left, upper, right, lower)
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        raise ValueError("Bounding box should be a list containing 4 elements (left, upper, right, lower).")
    print(bounding_box)
    # Crop the image based on the bounding box
    cropped_image = image.crop(bounding_box)

    transform_list = [
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_list)

    # Convert the cropped image to RGB and apply the transformations
    preprocessed_image = transform(cropped_image).unsqueeze(0)  # Add batch dimension

    return preprocessed_image


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


def calculate_similarity_faiss(query_vector, gallery_vectors):
    # Convert vectors to float32
    query_vector = np.array(query_vector, dtype=np.float32)
    gallery_vectors = np.array(gallery_vectors, dtype=np.float32)

    # Normalize the query vector
    query_vector /= np.linalg.norm(query_vector)

    # Normalize the gallery vectors
    gallery_vectors /= np.linalg.norm(gallery_vectors, axis=1, keepdims=True)

    # Initialize FAISS index
    index = faiss.IndexFlatIP(query_vector.shape[0])  # IndexFlatIP for inner product (cosine similarity)

    # Add gallery vectors to the index
    index.add(gallery_vectors)

    # Search for nearest neighbors
    num_neighbors = gallery_vectors.shape[0]
    top_k = 10  # Change this to the desired number of nearest neighbors
    D, I = index.search(query_vector.reshape(1, -1), top_k)

    # Convert similarity scores from inner product to cosine similarity
    similarity_scores = 0.5 + 0.5 * D.reshape(-1)

    return similarity_scores, I[0]
