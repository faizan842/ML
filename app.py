from flask import Flask, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load models
try:
    regressor = joblib.load('models/ridge_regression_model.pkl')
    feature_extractor = load_model('models/feature_extractor_model.h5')
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

# Function to extract features from a single image
def extract_features_from_image(image):
    try:
        img = cv2.resize(image, (256, 256))  # Resize the image to a consistent size
        img = np.expand_dims(img, axis=0)    # Add batch dimension
        img = img / 255.0  # Normalize image data
        features = feature_extractor.predict(img)  # Extract features using pre-trained CNN
        return features
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        raise e

# Function to crop the image
def crop_image(image):
    try:
        # Get image dimensions
        height, width, _ = image.shape

        # Define cropping margins
        left_margin = int(width * 0.444)
        right_margin = int(width * 0.558)
        top_margin = int(height * 0.49)
        bottom_margin = int(height * 0.541)

        # Crop the image
        cropped_image = image[top_margin:bottom_margin, left_margin:right_margin]

        return cropped_image
    except Exception as e:
        logging.error(f"Error cropping image: {e}")
        raise e

# Function to categorize vitamin D level
def categorize_vitamin_d_level(value):
    if value < 10:
        return "Severely Deficient", "extremely low"
    elif 10 <= value < 20:
        return "Deficient", "low"
    elif 20 <= value < 33:
        return "Insufficient", "slightly low"
    elif 33 <= value <= 100:
        return "Sufficient", "good"
    else:
        return "Consult a doctor", "Please consult a doctor for Vitamin D levels greater than 100 ng/mL"

# Route for predicting a single image
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image file from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        image_file = request.files['image']
        
        # Read the image
        image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Crop the image
        cropped_image = crop_image(image)

        # Extract features from the cropped image
        new_image_features = extract_features_from_image(cropped_image)

        # Use the trained Ridge regression model to predict the value
        predicted_value = regressor.predict(new_image_features)[0]  # Assuming the model returns a single value

        # Convert the predicted value to a standard Python float
        predicted_value = float(predicted_value)

        # Check for out-of-range predictions
        if predicted_value > 100 or predicted_value < 0:
            return jsonify({'error': 'Please reupload the image'}), 400

        # Categorize the predicted vitamin D level
        category, message = categorize_vitamin_d_level(predicted_value)

        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            'predicted_value': predicted_value, 
            'category': category, 
            'message': message, 
            'timestamp': current_time
        })
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return "Welcome to the Vitamin D Level Prediction API!"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(use_reloader=False, debug=True, host='0.0.0.0', port=port)
