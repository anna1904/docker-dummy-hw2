import logging
from typing import List

from serving.predictor import Predictor

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, image):
        results = self.predictor.predict(image)
        logger.info(results)
        return results
