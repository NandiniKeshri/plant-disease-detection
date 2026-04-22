from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
import os

app = Flask(__name__)

# Load model safely
model = tf.keras.models.load_model("plant_disease_model.h5")

# Load class names safely
class_names = sorted(os.listdir("dataset"))

# 🌿 Descriptions
descriptions = {
    "Tomato___Early_blight": "Fungal disease causing dark spots with concentric rings.",
    "Tomato___Late_blight": "Serious disease causing rapid decay of leaves and fruit.",
    "Potato___Early_blight": "Causes brown spots on leaves with target patterns.",
    "Potato___Late_blight": "Leads to rotting and blackened leaves.",
    "Pepper__bell__Bacterial_spot": "Causes water-soaked spots that turn brown."
}

# 💊 Remedies
remedies = {
    "Tomato___Early_blight": "Use fungicide and remove infected leaves.",
    "Tomato___Late_blight": "Apply copper fungicide and avoid moisture.",
    "Potato___Early_blight": "Use proper irrigation and resistant seeds.",
    "Potato___Late_blight": "Remove infected plants immediately.",
    "Pepper__bell__Bacterial_spot": "Use disease-free seeds and copper sprays."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No image selected")

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    result = class_names[class_index]

    description = descriptions.get(result, "General plant disease affecting leaf health.")
    remedy = remedies.get(result, "Maintain hygiene and monitor plant regularly.")

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        description=description,
        remedy=remedy
    )

# IMPORTANT for deployment
if __name__ == "__main__":
    # Render provides a PORT environment variable. We must use it.
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)