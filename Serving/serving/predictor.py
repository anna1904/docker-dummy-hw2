import logging
from pathlib import Path
import wandb
from filelock import FileLock
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer
from transformers import pipeline

logger = logging.getLogger()

MODEL_ID = "projector-team/classification-losses/classification-model:v0"
MODEL_PATH = "my_classification_model"
MODEL_LOCK = ".lock-file"
PROJECT_NAME = "classification-losses"


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
