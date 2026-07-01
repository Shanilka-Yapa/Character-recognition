from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)  # allows your frontend to securely communicate with this backend

# Setup base paths relative to this script's position
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'ml_models', 'saved_weights')

# 1. Load the Digit Model safely
digit_model_path = os.path.join(WEIGHTS_DIR, 'digits_model.h5')
if os.path.exists(digit_model_path):
    digit_model = tf.keras.models.load_model(digit_model_path)
    print("✅ Digits model loaded successfully!")
else:
    digit_model = None
    print("⚠️ Digits model file not found yet.")

# Placeholders for future expansions
english_model = None
sinhala_model = None

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    if 'file' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['file']
    
    try:
        # Preprocess image to standard 28x28 grayscale square
        img = Image.open(file.stream).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Route logic depending on which model the user selects on the UI
        if model_type == 'digits':
            if not digit_model:
                return jsonify({'error': 'Digit model not loaded on server'}), 500
            prediction = digit_model.predict(img_array)
            result = int(np.argmax(prediction, axis=1)[0])
            return jsonify({'prediction': str(result)})
            
        elif model_type == 'english':
            return jsonify({'prediction': 'English module placeholder'}), 200
            
        elif model_type == 'sinhala':
            return jsonify({'prediction': 'Sinhala module placeholder'}), 200
            
        else:
            return jsonify({'error': 'Unknown model type selection'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
