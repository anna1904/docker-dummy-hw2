import logging
from typing import List
import numpy as np
import time

from serving.predictor import Predictor
from PIL import Image

logger = logging.getLogger()


class Score:
    def __init__(self, num_classes):
        self.tp = np.zeros(num_classes)
        self.fp = np.zeros(num_classes)
        self.tn = np.zeros(num_classes)
        self.fn = np.zeros(num_classes)
        self.accuracy = 0
        self.precision = np.zeros(num_classes)


class SeldonAPI:
    def __init__(self):
        self.predictor = Predictor.default_from_model_registry()
        self._run_time = None
        self.num_classes = 10
        self.scores = Score(self.num_classes)

    def predict(self, image, features_names: List[str]):
        logger.info(image)

        s = time.perf_counter()
        np_array = np.array(image, dtype=np.uint8)
        image = Image.fromarray(np_array)
        results = self.predictor.predict(image)
        elapsed = time.perf_counter() - s
        self._run_time = elapsed

        logger.info(results)

        return results

    def metrics(self):
        precision = sum(self.scores.precision) / self.num_classes
        return [
            {"type": "GAUGE", "key": "gauge_runtime", "value": self._run_time},
            {"type": "GAUGE", "key": f"accuracy", "value": self.scores.accuracy},
            {"type": "GAUGE", "key": f"precision", "value": precision},
        ]

    def send_feedback(self, features, feature_names, reward, truth, routing=""):
        logger.info("features")
        logger.info(features)

        logger.info("truth")
        logger.info(truth)

        results = self.predict(features, feature_names)
        predicted_labels = [item['label'] for item in results]

        pred = predicted_labels[0]

        true_value = truth[0]

        if true_value == pred:
            self.scores.tp[int(true_value)] += 1
        else:
            self.scores.fp[pred] += 1
            self.scores.fn[int(true_value)] += 1

        self.scores.accuracy = sum(self.scores.tp) / (sum(self.scores.tp) + sum(self.scores.fp))

        for i in range(self.num_classes):
            if (self.scores.tp[i] + self.scores.fp[i]) != 0:
                self.scores.precision[i] = self.scores.tp[i] / (self.scores.tp[i] + self.scores.fp[i])
            else:
                self.scores.precision[i] = 0

        return []
