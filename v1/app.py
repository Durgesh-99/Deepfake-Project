import tensorflow as tf
from tensorflow import keras
import numpy as np
from flask import Flask, request, jsonify, make_response
from PIL import Image
from flask_cors import CORS
import io
import os

app = Flask(__name__)
CORS(app)

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.pos_embedding = self.add_weight(
            shape=(1, sequence_length, output_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        return inputs + tf.tile(self.pos_embedding, [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim
        })
        return config

# Load the model
custom_objects = {"PositionalEmbedding": PositionalEmbedding}
model = tf.keras.models.load_model("deepfake_proto1_bestweights_256_256.keras", custom_objects=custom_objects)

def preprocess_image(image):
    """Preprocess image to match model input requirements."""
    image = image.resize((256, 256))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET", "POST"])
def home_or_predict():
    if request.method == "GET":
        return make_response("Flask server is running!", 200)

    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Open image
        processed_image = preprocess_image(image)  # Preprocess

        predictions = model.predict(processed_image)  # Model inference
        predicted_class = int(np.argmax(predictions))  # Get class index
        confidence = float(np.max(predictions))  # Get confidence score
        return jsonify({"class": predicted_class, "confidence": confidence})
    
    return 0

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"🔥 Flask server is running on PORT={port} 🔥")
    app.run(host="0.0.0.0", port=port)