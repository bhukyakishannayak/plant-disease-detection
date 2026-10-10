"""Small web UI. Run with:  streamlit run app.py"""
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224


@st.cache_resource
def load():
    model = tf.keras.models.load_model("outputs/plant_disease_model.keras")
    with open("outputs/class_names.json") as f:
        return model, json.load(f)


st.title("Plant Disease Detection")
st.write("Upload a clear photo of a single leaf.")

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_container_width=True)
    model, class_names = load()
    arr = np.expand_dims(np.array(image.resize((IMG_SIZE, IMG_SIZE)), dtype="float32"), 0)
    probs = model.predict(arr)[0]
    for i in probs.argsort()[::-1][:3]:
        st.write(f"**{class_names[i]}**: {probs[i] * 100:.1f}%")
