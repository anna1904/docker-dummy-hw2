import logging
from typing import List
import base64
from io import BytesIO
import numpy as np

from serving.predictor import Predictor
from PIL import Image
import json

logger = logging.getLogger()


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()

    def predict(self, image, features_names: List[str]):
        # string = base64.b64decode(image)
        # image_io = BytesIO(string)
        # pil_image = Image.open(image_io)

        # logger.info(image)
        # data = json.loads(image)
        np_array = np.array(image, dtype=np.uint8)
        image = Image.fromarray(np_array)
        results = self.predictor.predict(image)
        # logger.info(results)
        return results
