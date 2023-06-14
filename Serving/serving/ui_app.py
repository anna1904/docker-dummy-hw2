import pandas as pd
import streamlit as st
from PIL import Image

from predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.default_from_model_registry()


predictor = get_model()

label_mapping = {
    '0_label': "Aircraft",
    '1_label': "Armoured_Fighting_Vehicles",
    '2_label': "Armoured_Personnel_Carriers",
    '3_label': "Artillery_Support_Vehicles_And_Equipment",
    '4_label': "Command_Posts_And_Communications_Stations",
    '5_label': "Engineering_Vehicles_And_Equipment",
    '6_label': "Helicopters",
    '7_label': "Infantry_Fighting_Vehicles",
    '8_label': "Infantry_Mobility_Vehicles",
    '9_label': "Mine-Resistant_Ambush_Protected",
    '10_label': "Multiple_Rocket_Launchers",
    '11_label': "Reconnaissance_Unmanned_Aerial_Vehicles",
    '12_label': "Self-Propelled_Anti-Tank_Missile_Systems",
    '13_label': "Self-Propelled_Artillery",
    '14_label': "Surface-To-Air_Missile_Systems",
    '15_label': "Tanks",
    '16_label': "Towed_Artillery",
    '17_label': "Trucks,_Vehicles_and_Jeeps"
}


def single_pred(label_mapping):
    image = st.file_uploader("Upload an image of broken vehicle", type=["jpg", "jpeg", "png"])
    if image is not None:
        pil_image = Image.open(image)
        st.image(pil_image, width=300)
    if st.button("Run prediction"):
        pred = predictor.predict(pil_image)
        for prediction in pred:
            label = prediction["label"]
            if label in label_mapping:
                prediction["label"] = label_mapping[label]

        formatted_predictions = []
        for prediction in pred:
            label = prediction["label"]
            score = prediction["score"]
            score_percent = int(score * 100)
            formatted_prediction = f"{label} ({score_percent}%)"
            formatted_predictions.append(formatted_prediction)

        st.write("RESULT => ", formatted_predictions)


def main():
    st.header("Image classificator for russian losses")

    single_pred(label_mapping)
    st.markdown(
        """
        <style>
        body {
            background-image: url('file:////Users/anko/Development/Projector/docker-dummy-hw2/Serving/serving/background.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
