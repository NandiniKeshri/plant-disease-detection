from flask import Flask, render_template, request
import numpy as np
import os
import gc
from PIL import Image
from pathlib import Path

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# ----------------------------
# LOAD MODEL LAZILY (IMPORTANT FOR RENDER)
# ----------------------------
model = None
interpreter = None
input_details = None
output_details = None
MODEL_DIR = Path(__file__).resolve().parent
TFLITE_MODEL_PATH = MODEL_DIR / "model.tflite"
KERAS_MODEL_PATH = MODEL_DIR / "plant_model.keras"

def load_tflite_once():
    global interpreter, input_details, output_details
    if interpreter is None:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter

        if not TFLITE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {TFLITE_MODEL_PATH.name}")

        interpreter = Interpreter(model_path=str(TFLITE_MODEL_PATH))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully")


def load_keras_once():
    global model
    if model is None:
        import tensorflow as tf

        if not KERAS_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {KERAS_MODEL_PATH.name}")

        model = tf.keras.models.load_model(str(KERAS_MODEL_PATH), compile=False)
        print("Keras model loaded successfully")


def predict_with_model(img):
    try:
        load_tflite_once()
        input_data = img.astype(input_details[0]["dtype"])
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])
    except Exception as tflite_error:
        print("TFLite prediction failed:", tflite_error)
        load_keras_once()
        return model.predict(img)

# ----------------------------
# CLASS NAMES (DO NOT USE os.listdir ON SERVER)
# ----------------------------
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
]

# ----------------------------
# DESCRIPTIONS
# ----------------------------
descriptions = {
    "Pepper__bell___Bacterial_spot": "Bacterial infection causing water-soaked spots that turn brown.",
    "Pepper__bell___healthy": "Healthy bell pepper leaf with no major disease symptoms detected.",
    "Potato___Early_blight": "Causes brown spots on leaves with target patterns.",
    "Potato___healthy": "Healthy potato leaf with no major disease symptoms detected.",
    "Potato___Late_blight": "Leads to rotting and blackened leaves.",
    "Tomato_Bacterial_spot": "Bacterial disease causing small dark lesions on tomato leaves and fruit.",
    "Tomato_Early_blight": "Fungal disease causing dark spots with concentric rings.",
    "Tomato_healthy": "Healthy tomato leaf with no major disease symptoms detected.",
    "Tomato_Late_blight": "Serious disease causing rapid decay of leaves and fruit.",
    "Tomato_Leaf_Mold": "Fungal disease that creates yellow leaf patches and mold on the underside.",
    "Tomato_Septoria_leaf_spot": "Fungal disease causing many small circular spots on older leaves.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Mite damage that causes stippling, yellowing, and webbing.",
    "Tomato__Target_Spot": "Fungal disease with target-like spots that can spread across leaves.",
    "Tomato__Tomato_mosaic_virus": "Viral disease causing mottled, distorted, or stunted tomato leaves.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Viral disease causing yellowing, curling leaves, and weak growth."
}

# ----------------------------
# REMEDIES
# ----------------------------
remedies = {
    "Pepper__bell___Bacterial_spot": "Use disease-free seeds, remove infected leaves, and apply copper sprays if recommended.",
    "Pepper__bell___healthy": "Keep regular watering, good sunlight, and routine monitoring.",
    "Potato___Early_blight": "Use proper irrigation and resistant seeds.",
    "Potato___healthy": "Maintain balanced nutrients, avoid overwatering, and inspect leaves weekly.",
    "Potato___Late_blight": "Remove infected plants immediately.",
    "Tomato_Bacterial_spot": "Remove infected parts, avoid overhead watering, and use copper-based control if suitable.",
    "Tomato_Early_blight": "Use fungicide and remove infected leaves.",
    "Tomato_healthy": "Continue preventive care and keep leaves dry during watering.",
    "Tomato_Late_blight": "Apply copper fungicide and avoid moisture.",
    "Tomato_Leaf_Mold": "Improve airflow, reduce humidity, and remove affected foliage.",
    "Tomato_Septoria_leaf_spot": "Remove lower infected leaves and mulch soil to reduce splash spread.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray leaves with water, use miticide if severe, and reduce plant stress.",
    "Tomato__Target_Spot": "Prune infected leaves, improve spacing, and apply a suitable fungicide.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants and disinfect tools to prevent spread.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and remove severely infected plants."
}

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    return {"status": "ok"}


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')

    if not file:
        return render_template('index.html', error="No image selected")

    if file.filename == '':
        return render_template('index.html', error="No image selected")

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return render_template(
            'index.html',
            error="Invalid image file. Please upload a clear JPG or PNG leaf photo."
        )

    # Preprocess
    image = image.resize((128, 128))
    img = np.asarray(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    try:
        prediction = predict_with_model(img)
    except Exception as e:
        print("Prediction failed:", e)
        return render_template(
            'index.html',
            error="Server could not load the ML model. Please check Render logs and model files."
        ), 500

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    if class_index >= len(class_names):
        return render_template(
            'index.html',
            error="Prediction class mismatch. Please check model and class names."
        ), 500

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
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
