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

# --- Path Setup ---
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / 'calibrated_grayscale_cnn.pth' 
UPLOAD_FOLDER_PATH = APP_DIR / 'uploads'

# --- Enhanced Startup Logging ---
env_path = APP_DIR / ".env"
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
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER_PATH) 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ========================================
# SOLUTION: MODEL ARCHITECTURE FROM TRAINING SCRIPT
# This is the definitive, correct architecture.
# ========================================

class AttentionModule(nn.Module):
    """Simple Spatial and Channel Attention Block from the training script."""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.avg_pool(x).squeeze(-1).squeeze(-1)
        ca = self.channel_attention(ca).unsqueeze(-1).unsqueeze(-1)
        x_ca = x * ca.expand_as(x)

        # Spatial Attention
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa.expand_as(x)

        return x_sa

class ResidualBlock(nn.Module):
    """A standard Residual Block from the training script."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AdvancedMedicalCNN(nn.Module):
    """The exact CNN architecture from the training script."""
    def __init__(self, num_classes=2):
        super(AdvancedMedicalCNN, self).__init__()
        # 1. Initial Convolution
        self.conv_initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 2. Residual and Attention Layers
        self.layer1 = self._make_layer(32, 64, 2)
        self.att1 = AttentionModule(64)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.att2 = AttentionModule(128)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # 3. Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Helper function to stack residual blocks, from training script."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_initial(x)
        x = self.layer1(x)
        x = self.att1(x)
        x = self.layer2(x)
        x = self.att2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------- Model Loading and Preprocessing ------------------

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Beginning', 'Malignant']

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def load_model_at_startup():
    global model
    logger.info(f"Checking for model file at: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at '{MODEL_PATH}'. Please ensure the model file is in the same directory as this script.")
        return

    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        num_classes = len(class_names)
        model = AdvancedMedicalCNN(num_classes=num_classes)
        
        state_dict = checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}", exc_info=True)
        model = None

# -------------- Helper and Prediction Functions ------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_single_image(image_path):
    """Predicts a single image, returns a dict with results or an error."""
    try:
        image = Image.open(image_path).convert('L')
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
        return jsonify({'error': 'No image files provided.'}), 400

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
            
    if supabase:
        first_valid_result = next((r for r in prediction_results if 'error' not in r), None)
        if first_valid_result:
            try:
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
                supabase.table('predictions').insert(record).execute()
            except Exception as e:
                logger.error(f"Database save failed: {e}", exc_info=True)

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

