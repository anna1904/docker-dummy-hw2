import numpy as np
import os
import random
from pathlib import Path
from locust import HttpUser, between, task

images_pool = [
    "tests/assets/img_10.png",
    "tests/assets/img_10.png",
]


class PredictUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        image_path = Path(os.getcwd()) / random.choice(images_pool)
        with open(image_path, 'rb') as image_file:
            response = self.client.post(
                "/predict",
                files={"image_file": ("filename", image_file, "image/jpeg")},
                headers={"accept": "application/json"}
            )
