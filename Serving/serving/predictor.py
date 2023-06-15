import logging
from pathlib import Path
import wandb
from filelock import FileLock
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer
from transformers import pipeline
import numpy as np
from PIL import Image
import base64
import requests
import json
import numpy
from io import BytesIO

logger = logging.getLogger()

MODEL_ID = "projector-team/classification_example/classification_example:v3"
MODEL_PATH = "my_classification_model"
MODEL_LOCK = ".lock-file"
PROJECT_NAME = "classification_example"


def load_from_registry(model_name: str, model_path: Path, project_name: str):
    with wandb.init(project_name) as run:
        artifact = run.use_artifact(model_name, type="model")
        artifact_dir = artifact.download(root=model_path)
        print(f"{artifact_dir}")


class Predictor:
    def __init__(self, model_load_path: str):
        self.image_processor = AutoImageProcessor.from_pretrained(model_load_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_load_path)
        self.classifier = pipeline("image-classification", model=model_load_path)

    def predict(self, image):
        return self.classifier(image)

    @classmethod
    def default_from_model_registry(cls) -> "Predictor":
        with FileLock(MODEL_LOCK):
            if not (Path(MODEL_PATH) / "pytorch_model.bin").exists():
                load_from_registry(model_name=MODEL_ID, model_path=MODEL_PATH, project_name=PROJECT_NAME)

        return cls(model_load_path=MODEL_PATH)

# predictor = Predictor.default_from_model_registry()

# with open("assets/img_10.png", "rb") as image_file:
#     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
#     data = {"data": {"ndarray": [encoded_string]}}
#     string = base64.b64decode(data['data']['ndarray'])
#     image_io = BytesIO(string)
#     pil_image = Image.open(image_io)
#
#     result = predictor.predict(pil_image)
#     print(result)
#
# np_img = np.random.randint(low=0, high=255, size=(5, 5), dtype=np.uint8)
# pil_img = Image.fromarray(np_img)
