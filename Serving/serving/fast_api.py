from typing import List, Dict

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image

from serving.predictor import Predictor


class Payload(BaseModel):
    image: UploadFile = File(...)


class Prediction(BaseModel):
    probs: List[Dict[str, float]]


app = FastAPI()
predictor = Predictor.default_from_model_registry()


@app.get("/health_check")
def health_check() -> str:
    return "ok"


@app.post("/predict", response_model=Prediction)
def predict(image: UploadFile = File(...)) -> Prediction:
    img = Image.open(image.file)
    prediction = predictor.predict(image=img)
    return Prediction(probs=prediction)
