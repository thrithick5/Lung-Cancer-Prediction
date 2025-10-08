import os
import io
import base64
import logging
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path

# --- Enhanced Startup Logging ---
# Load environment variables from a .env file
env_path = Path(__file__).with_name(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded environment variables from: {env_path}")
else:
    print(f"Warning: .env file not found at {env_path}. Relying on system environment variables.")

# --------------- Logging Setup ---------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log the status of environment variables immediately after loading
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
logger.info(f"SUPABASE_URL loaded: {bool(SUPABASE_URL)}")
logger.info(f"SUPABASE_KEY loaded: {bool(SUPABASE_KEY)}")


# ------------- Supabase Setup (Optional) ----------------
supabase = None
try:
    from supabase import create_client, Client
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    else:
        logger.warning("Supabase URL or Key is missing. Database logging will be disabled.")
except ImportError:
    logger.warning("Supabase client library not installed. Run 'pip install supabase' to enable database logging.")
except Exception as e:
    logger.error(f"An unexpected error occurred during Supabase initialization: {e}")


# --------------- Flask Setup -----------------
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================================
# ADVANCED MEDICAL CNN (MATCHING TRAINING SCRIPT)
# ========================================

class AdvancedMedicalCNN(nn.Module):
    """
    Advanced CNN architecture optimized for grayscale medical imaging.
    This MUST match the architecture from the training script.
    """
    def __init__(self, num_classes, dropout_rate=0.4):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res_block1 = self._make_residual_block(64, 128, stride=1)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        self.res_block4 = self._make_residual_block(512, 512, stride=2)
        self.spatial_attention = SpatialAttentionModule()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _make_residual_block(self, in_channels, out_channels, stride):
        return ResidualBlock(in_channels, out_channels, stride)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.spatial_attention(x)
        x = self.global_avg_pool(x)
        return self.classifier(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout2d(0.2)
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention

# -------------- Model Loading and Preprocessing ------------------

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model.pth'
class_names = ['Normal', 'Benign', 'Malignant'] # Default class names

# Preprocessing pipeline for Grayscale images (to match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def load_model_at_startup():
    global model, class_names
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the model is trained and the file is in the correct directory.")
        return

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        if 'class_names' in checkpoint and checkpoint['class_names']:
            class_names = checkpoint['class_names']
        else:
            logger.warning("Class names not found in checkpoint, using default.")
        
        num_classes = len(class_names)
        
        model = AdvancedMedicalCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully. Classes: {class_names}")

    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}", exc_info=True)
        model = None

# -------------- Helper and Prediction Functions ------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_single_image(image_path):
    """Predicts a single image, returns a dict with results or an error."""
    try:
        image = Image.open(image_path).convert('L') # Ensure image is grayscale
        tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
            result = {
                "predicted_class_name": class_names[predicted_idx.item()],
                "confidence_percentage": int(round(confidence.item() * 100))
            }
        return result
    except Exception as e:
        logger.error(f"Error during prediction for {image_path}: {e}", exc_info=True)
        return {"error": "Failed to process image."}

# --------------- API Routes ----------------------

@app.route('/')
def home():
    return render_template('frontpage.html')

@app.route('/detector')
def detector():
    return render_template('index.html')

@app.route('/model_status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded': model is not None,
        'message': 'Model ready' if model else 'Model not loaded',
        'device': str(device),
        'class_names': class_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded, please wait or check server logs.'}), 503

    if 'images' not in request.files:
        logger.warning(f"'/predict' called without 'images' key. Keys found: {list(request.files.keys())}")
        return jsonify({
            'error': 'No image files provided. Ensure files are sent under the key "images".',
            'received_keys': list(request.files.keys())
        }), 400

    files = request.files.getlist('images')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected for upload.'}), 400
    
    patient_info = {
        'patientName': request.form.get('patientName', 'Anonymous'),
        'patientAge': request.form.get('patientAge'),
        'patientGender': request.form.get('patientGender'),
        'smokingHistory': request.form.get('smokingHistory'),
    }
    
    prediction_results = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(path)
                result = predict_single_image(path)
                result['filename'] = filename
                prediction_results.append(result)
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}", exc_info=True)
                prediction_results.append({'filename': filename, 'error': 'Server failed to process this image.'})
            finally:
                if os.path.exists(path):
                    os.remove(path)
        elif file:
            prediction_results.append({'filename': file.filename, 'error': 'File type not allowed.'})
            
    # Log the first valid prediction to Supabase if available
    if supabase:
        first_valid_result = next((r for r in prediction_results if 'error' not in r), None)
        if first_valid_result:
            try:
                # *** SOLUTION: Add all patient info to the record ***
                record = {
                    'patient_name': patient_info['patientName'],
                    'patient_age': patient_info['patientAge'],
                    'patient_gender': patient_info['patientGender'],
                    'smoking_history': patient_info['smokingHistory'],
                    'prediction': first_valid_result.get('predicted_class_name'),
                    'confidence': first_valid_result.get('confidence_percentage'),
                    'model_version': 'AdvancedMedicalCNN_v1',
                    'prediction_timestamp': datetime.datetime.now().isoformat(),
                    'image_count': len(files)
                }
                logger.info(f"Attempting to insert record into Supabase: {record}")
                response = supabase.table('predictions').insert(record).execute()
                
                if response.data:
                    logger.info(f"Supabase insert successful. Response data: {response.data}")
                else:
                    logger.error(f"Supabase insert failed. Full response: {response}")

            except Exception as e:
                logger.error(f"Database save failed with an exception: {e}", exc_info=True)

    final_response = {
        'patient_info': patient_info,
        'results': prediction_results
    }
    
    return jsonify(final_response)

# --------------- Run App -----------------------
if __name__ == '__main__':
    logger.info("--- Starting Flask Application ---")
    load_model_at_startup()
    app.run(debug=True, host='0.0.0.0', port=5008)

