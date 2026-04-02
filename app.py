from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Lazy loading — model loads on first request, not at startup
model = None

def get_model():
    global model
    if model is None:
        import tensorflow as tf
        model = tf.keras.models.load_model("plant_disease_model.keras")
    return model

CONFIDENCE_THRESHOLD = 75

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

treatments = {
    "Apple scab": "Apply fungicide and remove infected leaves.",
    "Black rot": "Prune infected areas and use fungicide spray.",
    "Cedar apple rust": "Remove nearby cedar trees and apply fungicide.",
    "Powdery mildew": "Use sulfur-based fungicide and improve airflow.",
    "Early blight": "Use copper fungicide and remove infected leaves.",
    "Late blight": "Remove infected parts and apply systemic fungicide.",
    "Bacterial spot": "Use copper-based bactericide and avoid overhead watering.",
    "Leaf Mold": "Improve air circulation and reduce humidity.",
    "Septoria leaf spot": "Remove infected leaves and apply fungicide.",
    "Target Spot": "Use fungicide and improve plant spacing.",
    "healthy": "No disease detected. Your plant is healthy!"
}

@app.route("/")
def home():
    return "AgroVision Plant Disease Detection API is running!"

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        image = Image.open(file).convert("RGB")
        processed_image = preprocess_image(image)

        # Load model lazily
        m = get_model()

        prediction = m.predict(processed_image)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "status": "invalid",
                "message": "Invalid image. Please upload a clear plant leaf image.",
                "confidence": round(confidence, 2)
            })

        plant, disease = predicted_class.split("___")
        disease = disease.replace("_", " ")
        treatment = treatments.get(disease, "Consult agricultural expert for treatment.")

        return jsonify({
            "status": "success",
            "plant": plant,
            "disease": disease,
            "confidence": round(confidence, 2),
            "treatment": treatment
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)