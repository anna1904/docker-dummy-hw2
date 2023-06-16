import numpy as np
import os
import random
from pathlib import Path
from locust import HttpUser, between, task

images_pool = [
    "tests/assets/img_10.jpeg",
    "tests/assets/img_10.jpeg",
]


class PredictUser(HttpUser):
    wait_time = between(1, 50)

    @task
    def predict(self):
        image_path = random.choice(images_pool)
        with open(image_path, 'rb') as image_file:
            response = self.client.post(
                "/predict",
                files={'image': image_file},
                headers={"accept": "application/json"}
            )


a = PredictUser(HttpUser)
print(a.predict())
