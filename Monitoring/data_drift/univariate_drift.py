import os
import numpy as np
import datasets

from alibi_detect.cd import KernelDensityDrift

import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path


def extract_features(model, image):
    img = preprocess_image(image)
    return model(img).flatten().detach().numpy()


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# Example usage:
reference_folder = datasets.load_from_disk(
    Path('/Users/anko/Development/Projector/docker-dummy-hw2/classification/data/train.ds'))
test_image_path = "/Users/anko/Development/Projector/docker-dummy-hw2/Serving/tests/assets/img_10.jpeg"

# Load pre-trained classification model
model = models.resnet50(pretrained=True)
model.eval()

# Extract features from the reference dataset
reference_features = extract_features(model, reference_folder)

# Initialize and fit the KernelDensityDrift detector
detector = KernelDensityDrift(p_val=0.05)
detector.fit(reference_features)

# Load and preprocess the test image
test_image = Image.open(test_image_path).convert('RGB')

# Extract features from the test image
test_features = extract_features(model, test_image)

# Perform univariate drift detection using the trained detector
drift_predictions = detector.predict(test_features)

print("Univariate drift detected:", drift_predictions)
