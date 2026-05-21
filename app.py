from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os
import gc

app = Flask(__name__)

# ----------------------------
# LOAD MODEL ONLY ONCE (IMPORTANT FOR RENDER)
# ----------------------------
model = None

def load_model_once():
    global model
    if model is None:
        model = tf.keras.models.load_model(
    "plant_model.keras",
    compile=False
)

try:
    load_model_once()
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

# ----------------------------
# CLASS NAMES (DO NOT USE os.listdir ON SERVER)
# ----------------------------
class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Pepper__bell__Bacterial_spot"
]

# ----------------------------
# DESCRIPTIONS
# ----------------------------
descriptions = {
    "Tomato___Early_blight": "Fungal disease causing dark spots with concentric rings.",
    "Tomato___Late_blight": "Serious disease causing rapid decay of leaves and fruit.",
    "Potato___Early_blight": "Causes brown spots on leaves with target patterns.",
    "Potato___Late_blight": "Leads to rotting and blackened leaves.",
    "Pepper__bell__Bacterial_spot": "Causes water-soaked spots that turn brown."
}

# ----------------------------
# REMEDIES
# ----------------------------
remedies = {
    "Tomato___Early_blight": "Use fungicide and remove infected leaves.",
    "Tomato___Late_blight": "Apply copper fungicide and avoid moisture.",
    "Potato___Early_blight": "Use proper irrigation and resistant seeds.",
    "Potato___Late_blight": "Remove infected plants immediately.",
    "Pepper__bell__Bacterial_spot": "Use disease-free seeds and copper sprays."
}

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No image selected")

    # Read image safely
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Preprocess
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    result = class_names[class_index]

    description = descriptions.get(
        result,
        "General plant disease affecting leaf health."
    )

    remedy = remedies.get(
        result,
        "Maintain plant hygiene and monitor regularly."
    )

    # Cleanup memory (important for Render)
    gc.collect()

    return render_template(
        'index.html',
        result=result,
        confidence=round(confidence, 2),
        description=description,
        remedy=remedy
    )


# ----------------------------
# RUN (FOR LOCAL ONLY)
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0")