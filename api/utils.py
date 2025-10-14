# app/utils.py
from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image, target_size=(260,260)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def compare_embeddings(embedding1, embedding2, threshold=0.5):
    # placeholder para comparar embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold
