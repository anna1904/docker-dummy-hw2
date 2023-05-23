import logging

import datasets
from sklearn.model_selection import train_test_split
from transformers import pipeline

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoTokenizer
)

from pathlib import Path
import pandas as pd
from transformers import pipeline

from tqdm import tqdm

logger = logging.getLogger()


class Predictor:
    def __init__(self, model_load_path: str):
        self.image_processor = AutoImageProcessor.from_pretrained(model_load_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_load_path)
        self.classifier = pipeline("image-classification", model=model_load_path)

    def predict(self, image):
        return self.classifier(image)


def run_inference_on_ds(ds_path: Path, model_load_path: Path, result_path: Path):
    model = Predictor(model_load_path=model_load_path)
    dataset = datasets.load_from_disk(ds_path)
    correct_label = []
    for idx in range(len(dataset)):
        image = dataset[idx]['image']
        conf = model.predict(image)[0]['label']
        correct_label.append(conf)
    correct_label = pd.DataFrame(correct_label)
    correct_label.to_csv(result_path, index=False)

# run_inference_on_ds(Path('./../data/train.ds'), Path('./../my_classification_model'), Path('result/'))
