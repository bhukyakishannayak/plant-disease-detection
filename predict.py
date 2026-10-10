"""Predict the disease from a single leaf image.

Usage:  python predict.py path/to/leaf.jpg
"""
import json
import sys

import numpy as np
import tensorflow as tf

IMG_SIZE = 224

model = tf.keras.models.load_model("outputs/plant_disease_model.keras")
with open("outputs/class_names.json") as f:
    class_names = json.load(f)

img = tf.keras.utils.load_img(sys.argv[1], target_size=(IMG_SIZE, IMG_SIZE))
arr = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
probs = model.predict(arr)[0]

print("Top predictions:")
for i in probs.argsort()[::-1][:3]:
    print(f"  {class_names[i]:<55} {probs[i] * 100:.1f}%")
