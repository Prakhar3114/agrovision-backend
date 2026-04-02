from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
import io
import os

app = Flask(__name__)
CORS(app)

# TFLite interpreter (lazy loaded)
_interpreter = None

def get_interpreter():
    global _interpreter
    if _interpreter is None:
        import tflite_runtime.interpreter as tflite
        _interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
        _interpreter.allocate_tensors()
    return _interpreter

# 38 classes
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

TREATMENTS = {
    'Apple___Apple_scab': ['Apply fungicide spray every 7-10 days', 'Remove infected leaves immediately', 'Ensure proper air circulation around trees'],
    'Apple___Black_rot': ['Prune infected branches 15cm below visible damage', 'Apply copper-based fungicide', 'Remove mummified fruits from tree and ground'],
    'Apple___Cedar_apple_rust': ['Apply myclobutanil fungicide at bud break', 'Remove nearby juniper/cedar plants if possible', 'Rake and destroy fallen leaves'],
    'Apple___healthy': ['Continue regular watering schedule', 'Monitor for early signs of disease', 'Apply balanced fertilizer seasonally'],
    'Blueberry___healthy': ['Maintain soil pH between 4.5-5.5', 'Regular mulching recommended', 'Monitor for pest activity'],
    'Cherry_(including_sour)___Powdery_mildew': ['Apply sulfur-based fungicide', 'Improve air circulation by pruning', 'Avoid overhead irrigation'],
    'Cherry_(including_sour)___healthy': ['Regular pruning for air circulation', 'Monitor soil moisture levels', 'Apply dormant oil spray in winter'],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': ['Apply strobilurin fungicide', 'Rotate crops annually', 'Use resistant varieties next season'],
    'Corn_(maize)___Common_rust_': ['Apply propiconazole fungicide early', 'Plant rust-resistant hybrids', 'Monitor fields regularly after rainfall'],
    'Corn_(maize)___Northern_Leaf_Blight': ['Apply fungicide at first sign', 'Rotate with non-host crops', 'Till infected crop residue after harvest'],
    'Corn_(maize)___healthy': ['Maintain proper plant spacing', 'Ensure adequate nitrogen levels', 'Monitor for pest and disease signs'],
    'Grape___Black_rot': ['Apply mancozeb or myclobutanil fungicide', 'Remove all infected berries and leaves', 'Ensure good canopy ventilation'],
    'Grape___Esca_(Black_Measles)': ['No cure available - manage by pruning', 'Apply wound sealant after pruning', 'Reduce vine stress through irrigation'],
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': ['Apply copper fungicide', 'Remove infected leaves', 'Improve drainage around vines'],
    'Grape___healthy': ['Regular canopy management', 'Monitor irrigation levels', 'Apply preventive fungicide before rain season'],
    'Orange___Haunglongbing_(Citrus_greening)': ['No cure - remove infected trees immediately', 'Control Asian citrus psyllid with insecticide', 'Use certified disease-free planting material'],
    'Peach___Bacterial_spot': ['Apply copper bactericide', 'Avoid overhead watering', 'Prune to improve air circulation'],
    'Peach___healthy': ['Apply dormant copper spray in fall', 'Thin fruit for better airflow', 'Monitor for brown rot signs'],
    'Pepper,_bell___Bacterial_spot': ['Apply copper-based bactericide every 5-7 days', 'Remove infected plant material', 'Avoid working in wet conditions'],
    'Pepper,_bell___healthy': ['Maintain consistent soil moisture', 'Apply balanced NPK fertilizer', 'Monitor for aphid activity'],
    'Potato___Early_blight': ['Apply chlorothalonil or mancozeb fungicide', 'Remove lower infected leaves', 'Ensure adequate potassium nutrition'],
    'Potato___Late_blight': ['Apply metalaxyl fungicide immediately', 'Destroy infected plant material', 'Harvest early if infection is severe'],
    'Potato___healthy': ['Hill soil around plants regularly', 'Monitor irrigation - avoid waterlogging', 'Apply preventive fungicide in humid weather'],
    'Raspberry___healthy': ['Prune old canes after fruiting', 'Maintain proper row spacing', 'Apply mulch to retain moisture'],
    'Soybean___healthy': ['Monitor for soybean cyst nematode', 'Maintain proper soil pH 6.0-6.5', 'Rotate with non-legume crops'],
    'Squash___Powdery_mildew': ['Apply potassium bicarbonate spray', 'Improve air circulation', 'Water at base of plant only'],
    'Strawberry___Leaf_scorch': ['Apply captan fungicide', 'Remove infected leaves', 'Avoid overhead watering'],
    'Strawberry___healthy': ['Renovate beds after harvest', 'Control runners to prevent overcrowding', 'Apply balanced fertilizer after fruiting'],
    'Tomato___Bacterial_spot': ['Apply copper bactericide + mancozeb mix', 'Remove infected leaves immediately', 'Avoid working with wet plants'],
    'Tomato___Early_blight': ['Apply chlorothalonil fungicide', 'Mulch around base of plants', 'Remove lower leaves touching soil'],
    'Tomato___Late_blight': ['Apply mancozeb or copper fungicide immediately', 'Remove all infected plant material', 'Avoid overhead irrigation'],
    'Tomato___Leaf_Mold': ['Improve greenhouse ventilation', 'Apply copper-based fungicide', 'Reduce humidity below 85%'],
    'Tomato___Septoria_leaf_spot': ['Apply fungicide containing chlorothalonil', 'Remove infected lower leaves', 'Stake plants for better airflow'],
    'Tomato___Spider_mites Two-spotted_spider_mite': ['Apply miticide or insecticidal soap', 'Spray water on undersides of leaves', 'Introduce predatory mites as biocontrol'],
    'Tomato___Target_Spot': ['Apply azoxystrobin fungicide', 'Ensure proper plant spacing', 'Remove plant debris after harvest'],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': ['No cure - remove infected plants immediately', 'Control whitefly vectors with insecticide', 'Use reflective mulch to deter whiteflies'],
    'Tomato___Tomato_mosaic_virus': ['No cure - remove infected plants', 'Disinfect tools with bleach solution', 'Wash hands thoroughly after handling plants'],
    'Tomato___healthy': ['Maintain consistent watering schedule', 'Apply calcium to prevent blossom end rot', 'Monitor for early pest signs'],
}

CONFIDENCE_THRESHOLD = 0.5

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return jsonify({'message': 'AgroVision Plant Disease Detection API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        image_data = base64.b64decode(data['image'])
        img_array = preprocess_image(image_data)

        interpreter = get_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                'status': 'invalid',
                'message': 'Image does not appear to be a plant leaf',
                'confidence': round(confidence * 100, 2)
            })

        predicted_class = CLASS_NAMES[predicted_index]
        plant, condition = predicted_class.split('___')
        is_healthy = 'healthy' in condition.lower()
        treatment = TREATMENTS.get(predicted_class, ['Consult an agricultural expert'])

        all_confidences = {CLASS_NAMES[i]: round(float(predictions[i]) * 100, 2) for i in range(len(CLASS_NAMES))}

        return jsonify({
            'status': 'success',
            'plant': plant.replace('_', ' '),
            'disease': condition.replace('_', ' '),
            'is_healthy': is_healthy,
            'confidence': round(confidence * 100, 2),
            'all_confidences': all_confidences,
            'treatment': treatment
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)