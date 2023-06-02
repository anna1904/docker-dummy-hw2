import logging
from typing import List

from serving.predictor import Predictor
from PIL import Image

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, image):
        img = Image.open(image)
        results = self.predictor.predict(img)
        logger.info(results)
        return results
