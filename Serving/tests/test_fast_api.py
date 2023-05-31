import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from serving.fast_api import app

import io
from PIL import Image

client = TestClient(app)


def test_health_check():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == "ok"


def test_predict():
    # Prepare a sample image
    image_data = io.BytesIO()
    Image.new("RGB", (100, 100)).save(image_data, format="PNG")
    image_data.seek(0)

    response = client.post("/predict", files={"image": ("image.png", image_data, "image/png")})

    assert response.status_code == 200
    response_data = response.json()
    assert "probs" in response_data
    assert isinstance(response_data["probs"], list)
    for item in response_data["probs"]:
        assert isinstance(item, dict)
        assert "label" in item
        assert "score" in item
        assert isinstance(item["label"], float)
        assert isinstance(item["score"], float)
