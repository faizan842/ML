from flask import Flask, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import logging
from datetime import datetime
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load models
try:
    regressor = joblib.load('models/ridge_regression_model.pkl')
    feature_extractor = load_model('models/feature_extractor_model.h5')
    model = YOLO('yolov8-vitamin-d-kit.pt')
    logging.info("Models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise e

# AWS S3 configuration
S3_BUCKET = os.getenv('S3_BUCKET')
S3_REGION = os.getenv('S3_REGION')
S3_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
S3_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

s3 = boto3.client('s3',
                  region_name=S3_REGION,
                  aws_access_key_id=S3_ACCESS_KEY,
                  aws_secret_access_key=S3_SECRET_KEY)

# Global variable to store the previous predicted value
previous_prediction = None

# Function to extract features from a single image
def extract_features_from_image(image):
    try:
        img = cv2.resize(image, (128, 128))  # Resize the image to the expected input size
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
    elif 33 <= value <= 110:
        return "Sufficient", "good"
    else:
        return "Consult a doctor", "Please consult a doctor for Vitamin D levels greater than 110 ng/mL"

# Function to upload image to S3
def upload_to_s3(file_content, bucket_name, s3_filename):
    try:
        s3.put_object(Bucket=bucket_name, Key=s3_filename, Body=file_content)
        logging.info(f"File uploaded to S3: {s3_filename}")
        return f"https://{bucket_name}.s3.{S3_REGION}.amazonaws.com/{s3_filename}"
    except NoCredentialsError:
        logging.error("Credentials not available")
        raise
    except ClientError as e:
        logging.error(f"Client error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error uploading to S3: {e}")
        raise

def auto_crop(image):
    try:
        # Predict using YOLOv8 model
        results = model.predict(image, imgsz=640, conf=0.1)
        
        # Check if any detections were made
        if results and len(results[0].boxes) > 0:
            # Get the first result (assuming single object detection)
            result = results[0]
            box = result.boxes[0].xyxy[0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            # Crop the image using the bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]
            return cropped_image
        else:
            logging.info("No detections were made.")
            return image  # Return the original image if no detections
    except Exception as e:
        logging.error(f"Error during auto-cropping: {e}")
        raise e

# Function to calculate SSIM between two images
def calculate_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# Resize images to the same dimensions
def resize_images(images, size):
    resized_images = [cv2.resize(image, size) for image in images]
    return resized_images

# Route for predicting from multiple images
# Route for predicting from multiple images
@app.route('/predict', methods=['POST'])
def predict_multiple():
    global previous_prediction
    
    try:
        # Get image files from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        image_files = request.files.getlist('image')

        if len(image_files) < 1:
            return jsonify({'error': 'Please provide at least 2 images'}), 400

        # Read the images
        images = [cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR) for image_file in image_files]
        if any(image is None for image in images):
            return jsonify({'error': 'Invalid image file(s)'}), 400

        # Determine the common size (e.g., the size of the first image)
        common_size = (images[0].shape[1], images[0].shape[0])

        # Resize images to the common size
        resized_images = resize_images(images, common_size)

        # Calculate average SSIM for each image
        average_ssim_values = []
        for i in range(len(resized_images)):
            ssim_scores = []
            for j in range(len(resized_images)):
                if i != j:
                    ssim_scores.append(calculate_ssim(resized_images[i], resized_images[j]))
            average_ssim = np.mean(ssim_scores)
            average_ssim_values.append(average_ssim)

        # Find the most stable image
        most_stable_image_index = np.argmin(average_ssim_values)
        most_stable_image = images[most_stable_image_index]

        # Auto crop the image
        image = auto_crop(most_stable_image)

        # Crop the image again (second crop)
        cropped_image = crop_image(image)

        # Extract features from the cropped image
        new_image_features = extract_features_from_image(cropped_image)

        # Use the trained Ridge regression model to predict the value
        predicted_value = regressor.predict(new_image_features)[0]  # Assuming the model returns a single value

        # Round the predicted value to 1 decimal place
        predicted_value_rounded = round(float(predicted_value), 1)

        # Check for out-of-range predictions
        if predicted_value_rounded > 110 or predicted_value_rounded < 0:
            return jsonify({'error': 'Please upload clear image'}), 400

        # Compare with the previous prediction
        if previous_prediction is not None:
            if abs(predicted_value_rounded - previous_prediction) <= 8 and predicted_value_rounded > 50:
                predicted_value_rounded = previous_prediction
            elif abs(predicted_value_rounded - previous_prediction) <= 6:
                predicted_value_rounded = previous_prediction
            else:
                previous_prediction = predicted_value_rounded
        else:
            previous_prediction = predicted_value_rounded

        # Categorize the predicted vitamin D level
        category, message = categorize_vitamin_d_level(predicted_value_rounded)

        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Generate a unique filename based on predicted value and current timestamp
        unique_filename = f"{predicted_value_rounded}_{current_time}.jpg"

        # Save the image to S3 with the unique filename
        _, image_buffer = cv2.imencode('.jpg', image)
        s3_url = upload_to_s3(image_buffer.tobytes(), S3_BUCKET, unique_filename)

        return jsonify({
            'predicted_value': predicted_value_rounded, 
            'category': category, 
            'message': message, 
            'timestamp': current_time,
            's3_url': s3_url
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
