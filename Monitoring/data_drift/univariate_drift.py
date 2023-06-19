import os
import numpy as np
import datasets

from alibi_detect.cd import KSDrift

import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from alibi_detect.saving import save_detector


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


def detect_drift(dataset_path, test_image_path, detector_path):
    reference_dataset = datasets.load_from_disk(dataset_path)

    model = models.resnet50(pretrained=True)
    model.eval()

    reference_features = []

    for data in reference_dataset:
        image = data['image'].convert("RGB")
        features = extract_features(model, image)
        reference_features.append(features)

    reference_features = np.array(reference_features)
    detector = KSDrift(x_ref=reference_features, p_val=0.05)
    save_detector(detector, detector_path)
    test_image = Image.open(test_image_path).convert('RGB')

    test_features = extract_features(model, test_image).reshape(1, -1)
    drift_predictions = detector.predict(test_features, return_p_val=True)

    print(drift_predictions['data']['is_drift'])
    print(drift_predictions['data']['p_val'])
