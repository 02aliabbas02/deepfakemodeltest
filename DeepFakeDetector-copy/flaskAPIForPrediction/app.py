import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from icpr_model import ICPRModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Instantiate your ICPR model
model = ICPRModel()

def make_prediction(image_file):
    try:
        # Make prediction using ICPR model
        prediction = model.predict(image_file)
        return prediction
    except Exception as e:
        return {"error": str(e)}

@app.route('/predict_icpr', methods=['POST'])
def predict_icpr():
    app.logger.info('Predict endpoint called')
    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    app.logger.info(f'File received: {file.filename}')
    
    if file.filename == '':
        app.logger.error('No selected file')
        return jsonify({"error": "No selected file"}), 400
    
    try:
        prediction = make_prediction(file)
        app.logger.info(f'Prediction: {prediction}')
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return jsonify({"message": "Welcome to ICPR 2020 DFDC model integration!"})

if __name__ == "__main__":
    app.run(debug=True)
