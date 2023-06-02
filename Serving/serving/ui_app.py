import pandas as pd
import streamlit as st
from PIL import Image

from predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.default_from_model_registry()


predictor = get_model()


def single_pred():
    image = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
    if image is not None:
        pil_image = Image.open(image)
    if st.button("Run inference"):
        st.write("Input:", image)
        pred = predictor.predict(pil_image)
        st.write("Pred:", pred)


def main():
    st.header("UI serving demo")

    single_pred()


if __name__ == "__main__":
    main()
