import tensorflow as tf

# Load old model
model = tf.keras.models.load_model(
    "plant_disease_model.h5",
    compile=False
)

# Save in new format
model.save("plant_model.keras")

print("Model converted successfully")