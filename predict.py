import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("plant_disease_model.h5")

# Load class labels
class_names = sorted(os.listdir("dataset"))

# Load test image
img_path = "test.jpg"   # 👉 change this to your test image name
img = cv2.imread(img_path)

img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)

print("Prediction:", class_names[class_index])