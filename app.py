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

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# --------------- Logging Setup ---------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"SUPABASE_URL present: {bool(os.getenv('SUPABASE_URL'))}")
logger.info(f"SUPABASE_KEY present: {bool(os.getenv('SUPABASE_KEY'))}")

# ------------- Supabase Setup ----------------
supabase = None
try:
    from supabase import create_client, Client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    from typing import Optional

    supabase: Optional[Client] = None
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase configuration not found in environment variables")
    else:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase initialized")

        # Quick connectivity + permissions smoke test
        try:
            test = supabase.table("predictions").select("id").limit(1).execute()
            logger.info("Supabase query ok")
        except Exception as e:
            logger.error(f"Supabase query failed: {e}")
            logger.info("Check that the 'predictions' table exists and RLS policies allow this action.")
            
except ImportError:
    logger.error("Supabase client not installed. Run: pip install supabase")
except Exception as e:
    logger.error(f"Supabase initialization failed: {e}")

# --------------- Flask Setup -----------------
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------------- Enhanced Model Class with Uncertainty ------------------
class EnhancedMedicalCNN(nn.Module):
    """Enhanced CNN model with Monte Carlo Dropout for uncertainty estimation"""
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )
        
        # Calculate the size after conv layers
        self.feature_size = 256 * (224 // 16) * (224 // 16)  # 224/16 = 14
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def mc_dropout_predict(self, x, n_samples=20):
        """Perform Monte Carlo Dropout prediction for uncertainty estimation"""
        self.train()  # Enable dropout during inference
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x)
                predictions.append(F.softmax(output, dim=1))
        
        self.eval()  # Return to eval mode
        return torch.stack(predictions)

# -------------- Helper function for JSON serialization ------------------
def convert_to_json_serializable(obj):
    """Convert numpy/torch types to JSON serializable Python types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# -------------- Improved Prediction Functions ------------------
def calculate_prediction_uncertainty(model, image_tensor, device, n_samples=20):
    """Calculate prediction uncertainty using Monte Carlo Dropout"""
    model.eval()
    
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Get multiple predictions with dropout enabled
    mc_predictions = model.mc_dropout_predict(image_tensor, n_samples)
    
    # Calculate statistics
    mean_pred = torch.mean(mc_predictions, dim=0).squeeze()
    std_pred = torch.std(mc_predictions, dim=0).squeeze()
    
    # Predictive entropy (uncertainty measure)
    entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8))
    
    # Max prediction confidence
    max_confidence = torch.max(mean_pred)
    
    return {
        'mean_probabilities': mean_pred.cpu().numpy(),
        'std_probabilities': std_pred.cpu().numpy(),
        'predictive_entropy': float(entropy.item()),  # Convert to Python float
        'max_confidence': float(max_confidence.item())  # Convert to Python float
    }

def interpret_medical_prediction(probabilities, std_probabilities, entropy, class_names, confidence_threshold=0.7):
    """
    Provide medically appropriate interpretation without artificial mapping
    """
    prob_percentages = probabilities * 100
    std_percentages = std_probabilities * 100
    
    # Find predicted class
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    confidence = prob_percentages[predicted_idx]
    uncertainty = std_percentages[predicted_idx]
    
    # Determine reliability based on confidence and uncertainty
    if confidence >= confidence_threshold * 100 and entropy < 0.5:
        reliability = "High"
        reliability_note = "Model shows consistent predictions with low uncertainty"
    elif confidence >= 50 and entropy < 1.0:
        reliability = "Moderate"
        reliability_note = "Model prediction has moderate confidence with some uncertainty"
    else:
        reliability = "Low"
        reliability_note = "Model prediction has low confidence or high uncertainty - requires careful review"
    
    # Medical risk categorization based on actual probabilities
    risk_assessment = categorize_medical_risk(predicted_class, confidence, reliability)
    
    # Prepare detailed class probabilities with uncertainties
    detailed_probabilities = []
    for i, class_name in enumerate(class_names):
        detailed_probabilities.append({
            'class': class_name,
            'probability': float(round(prob_percentages[i], 2)),  # Convert to Python float
            'uncertainty': float(round(std_percentages[i], 2)),  # Convert to Python float
            'confidence_interval': f"{max(0, prob_percentages[i] - 1.96*std_percentages[i]):.1f}% - {min(100, prob_percentages[i] + 1.96*std_percentages[i]):.1f}%"
        })
    
    return {
        'predicted_class': predicted_class,
        'model_confidence': float(round(confidence, 2)),  # Convert to Python float
        'prediction_uncertainty': float(round(uncertainty, 2)),  # Convert to Python float
        'predictive_entropy': float(round(entropy, 3)),  # Convert to Python float
        'reliability': reliability,
        'reliability_note': reliability_note,
        'risk_assessment': risk_assessment,
        'detailed_probabilities': detailed_probabilities,
        'medical_disclaimer': "This AI analysis is for reference only and must be reviewed by qualified medical professionals. Do not use for clinical decision making without proper medical consultation."
    }

def categorize_medical_risk(predicted_class, confidence, reliability):
    """Categorize medical risk based on prediction and reliability"""
    class_lower = predicted_class.lower()
    
    if class_lower == 'normal':
        if reliability == "High" and confidence > 80:
            return {
                'level': 'Low concern for abnormality',
                'recommendation': 'Continue routine screening as recommended by healthcare provider',
                'urgency': 'Routine follow-up'
            }
        else:
            return {
                'level': 'Normal findings with uncertainty',
                'recommendation': 'Consider additional imaging or clinical correlation if symptoms persist',
                'urgency': 'Clinical correlation recommended'
            }
    
    elif class_lower in ['benign', 'beging']:
        if reliability == "High" and confidence > 70:
            return {
                'level': 'Likely benign finding',
                'recommendation': 'Follow-up imaging and clinical evaluation recommended',
                'urgency': 'Non-urgent medical consultation'
            }
        else:
            return {
                'level': 'Possible benign finding with uncertainty',
                'recommendation': 'Additional imaging and specialist consultation recommended',
                'urgency': 'Medical evaluation within reasonable timeframe'
            }
    
    elif class_lower == 'malignant':
        return {
            'level': 'Concerning finding requiring immediate attention',
            'recommendation': 'Urgent specialist consultation and additional diagnostic workup required',
            'urgency': 'Immediate medical attention - do not delay'
        }
    
    else:
        return {
            'level': 'Unclear finding',
            'recommendation': 'Clinical correlation and additional imaging recommended',
            'urgency': 'Medical evaluation recommended'
        }

def predict_ct_scan_with_uncertainty(model, image_tensor, class_names, device):
    """Make prediction with proper uncertainty quantification"""
    try:
        # Calculate uncertainty
        uncertainty_results = calculate_prediction_uncertainty(model, image_tensor, device)
        
        # Get interpretation
        result = interpret_medical_prediction(
            uncertainty_results['mean_probabilities'],
            uncertainty_results['std_probabilities'],
            uncertainty_results['predictive_entropy'],
            class_names
        )
        
        # Ensure all values are JSON serializable
        result = convert_to_json_serializable(result)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in uncertainty prediction: {e}")
        raise

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_lung_cancer_cnn_enhanced.pth'

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class_names = ['Normal', 'Benign', 'Malignant']

def load_model_at_startup():
    global model, class_names
    
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Loading Enhanced Medical CNN model from: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            saved_class_names = checkpoint.get('class_names', ['Normal', 'Benign', 'Malignant'])
            
            # Clean class names
            saved_class_names = [name.strip() for name in saved_class_names]
            num_classes = len(saved_class_names)
            
            # Load enhanced medical CNN model
            logger.info("Loading enhanced medical CNN model architecture")
            model = EnhancedMedicalCNN(num_classes)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            class_names = saved_class_names
            logger.info(f"Enhanced medical model loaded successfully with {num_classes} classes: {class_names}")
            logger.info(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to load enhanced model from {MODEL_PATH}: {e}")
            model = None
    else:
        logger.warning(f"Enhanced model file not found: {MODEL_PATH}. Please train the model first.")

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def safe_filename(filename):
    """Generate a safe filename to prevent path traversal attacks"""
    import uuid
    import time
    timestamp = str(int(time.time()))
    random_id = str(uuid.uuid4())[:8]
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
    return f"{timestamp}_{random_id}.{extension}"

def predict_image(img_path):
    """Enhanced prediction function with uncertainty quantification"""
    try:
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img)
        
        # Use the enhanced prediction function with uncertainty
        result = predict_ct_scan_with_uncertainty(model, tensor, class_names, device)
        
        return result
    except Exception as e:
        logger.error(f"Error in predict_image: {e}")
        raise

# --------------- Routes ----------------------

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
        'message': 'Enhanced Medical CNN model ready' if model else 'Model not loaded',
        'supabase_connected': supabase is not None,
        'device': str(device),
        'class_names': class_names,
        'model_type': 'Enhanced Medical CNN with Uncertainty Quantification'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400

        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400

        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        # Get patient info with improved validation
        patient_info = {}
        
        # Extract and validate patient name
        patient_name = request.form.get('patientName', '').strip()
        if not patient_name:
            return jsonify({'error': 'Patient name is required'}), 400
        if len(patient_name) < 2:
            return jsonify({'error': 'Patient name must be at least 2 characters'}), 400
        patient_info['patientName'] = patient_name

        # Extract and validate age
        try:
            age_str = request.form.get('patientAge', '').strip()
            if not age_str:
                return jsonify({'error': 'Patient age is required'}), 400
            age = int(age_str)
            if age < 1 or age > 120:
                return jsonify({'error': 'Age must be between 1 and 120'}), 400
            patient_info['patientAge'] = age
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid age format'}), 400

        # Extract and validate gender
        gender = request.form.get('patientGender', '').strip()
        if not gender or gender not in ['Male', 'Female', 'Other']:
            return jsonify({'error': 'Valid gender selection is required'}), 400
        patient_info['patientGender'] = gender

        # Extract and validate smoking history
        smoking_history = request.form.get('smokingHistory', '').strip()
        valid_smoking_options = ['Never Smoked', 'Former Smoker', 'Current Smoker']
        if not smoking_history or smoking_history not in valid_smoking_options:
            return jsonify({'error': 'Valid smoking history selection is required'}), 400
        patient_info['smokingHistory'] = smoking_history

        logger.info(f"Processing request for patient: {patient_info}")

        results = []

        for f in files:
            if f and f.filename and allowed_file(f.filename):
                safe_name = safe_filename(f.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
                
                try:
                    f.save(path)
                    
                    # Verify file was saved and is readable
                    if not os.path.exists(path):
                        raise Exception("File was not saved properly")
                    
                    # Check file size
                    file_size = os.path.getsize(path)
                    if file_size == 0:
                        raise Exception("Saved file is empty")
                    
                    logger.info(f"Processing image: {f.filename} (saved as {safe_name})")
                    
                    # Make prediction with uncertainty
                    result = predict_image(path)
                    
                    # Format result for API response - ensure JSON serialization
                    prediction_result = convert_to_json_serializable({
                        'filename': f.filename,
                        'predicted_class': result['predicted_class'],
                        'predicted_class_name': result['predicted_class'],  # Add this for frontend compatibility
                        'confidence_percentage': result['model_confidence'],  # Add this for frontend compatibility
                        'model_confidence': result['model_confidence'],
                        'prediction_uncertainty': result['prediction_uncertainty'],
                        'predictive_entropy': result['predictive_entropy'],
                        'reliability': result['reliability'],
                        'reliability_note': result['reliability_note'],
                        'risk_assessment': result['risk_assessment'],
                        'detailed_probabilities': result['detailed_probabilities'],
                        'medical_disclaimer': result['medical_disclaimer']
                    })

                    results.append(prediction_result)
                    logger.info(f"Successful prediction for {f.filename}: {result['predicted_class']} (Confidence: {result['model_confidence']}%, Reliability: {result['reliability']})")
                    
                except Exception as e:
                    logger.error(f"Prediction error for {f.filename}: {e}")
                    results.append({
                        'filename': f.filename, 
                        'error': f'Prediction failed: {str(e)}'
                    })
                finally:
                    # Clean up file
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to cleanup file {path}: {cleanup_error}")
            else:
                if f.filename:
                    results.append({
                        'filename': f.filename,
                        'error': 'Invalid file type. Only PNG, JPG, JPEG allowed.'
                    })

        if not results:
            return jsonify({'error': 'No valid images processed'}), 400

        if supabase:
            try:
                record = {
                    'patient_name': patient_info['patientName'],
                    'patient_age': patient_info['patientAge'],
                    'patient_gender': patient_info['patientGender'],
                    'smoking_history': patient_info['smokingHistory'],
                    'analysis_results': results,
                    'model_version': 'Enhanced Medical CNN with Uncertainty Quantification',
                    'prediction_timestamp': datetime.datetime.now().isoformat()
                }
                
                logger.info(f"Saving medical record to database: {record}")
                
                response = supabase.table('predictions').insert(record).execute()
                
                if response.data:
                    logger.info(f"Medical record saved successfully with ID: {response.data[0].get('id')}")
                else:
                    logger.error("No data returned from database insert")
                    
            except Exception as e:
                logger.error(f"Database save failed: {e}")
                logger.warning("Continuing without database save")
        else:
            logger.warning("Supabase not available - record not saved to database")

        # Ensure the final response is JSON serializable
        response_data = convert_to_json_serializable({
            'results': results, 
            'patient_info': patient_info,
            'model_type': 'Enhanced Medical CNN with Uncertainty Quantification',
            'important_notice': 'All predictions must be reviewed by qualified medical professionals. This AI system is designed to assist, not replace, clinical judgment.'
        })

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        return jsonify({
            'error': f'Server error: {str(e)}',
            'details': 'Check server logs for more information'
        }), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Endpoint for single image prediction with detailed medical output"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if not (file and file.filename and allowed_file(file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400

        safe_name = safe_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        file.save(path)

        try:
            # Get enhanced prediction with uncertainty
            result = predict_image(path)
            
            # Format medical response - ensure JSON serialization
            response = convert_to_json_serializable({
                'filename': file.filename,
                'predicted_class': result['predicted_class'],
                'model_confidence': result['model_confidence'],
                'prediction_uncertainty': result['prediction_uncertainty'],
                'reliability': result['reliability'],
                'risk_assessment': result['risk_assessment'],
                'detailed_probabilities': result['detailed_probabilities'],
                'medical_disclaimer': result['medical_disclaimer']
            })
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Prediction error for {file.filename}: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(path):
                os.remove(path)

    except Exception as e:
        logger.error(f"Unexpected error in predict_single endpoint: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/model_calibration', methods=['GET'])
def get_model_calibration():
    """Endpoint to check model calibration on validation data"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 400

    try:
        val_dir = 'processed_dataset/val'
        if not os.path.exists(val_dir):
            return jsonify({'error': 'Validation dataset not found'}), 400
            
        ds = datasets.ImageFolder(val_dir, transform=preprocess)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        all_probs = []
        all_labels = []
        all_uncertainties = []

        model.eval()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                
                # Get uncertainty estimates
                mc_predictions = model.mc_dropout_predict(imgs, n_samples=10)
                mean_pred = torch.mean(mc_predictions, dim=0)
                std_pred = torch.std(mc_predictions, dim=0)
                
                all_probs.extend(mean_pred.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_uncertainties.extend(torch.mean(std_pred, dim=1).cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        all_uncertainties = np.array(all_uncertainties)
        
        # Calculate calibration for each class
        calibration_results = {}
        for class_idx, class_name in enumerate(class_names):
            binary_labels = (all_labels == class_idx).astype(int)
            class_probs = all_probs[:, class_idx]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    binary_labels, class_probs, n_bins=10
                )
                calibration_results[class_name] = {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist()
                }
            except Exception as e:
                logger.warning(f"Calibration calculation failed for {class_name}: {e}")
        
        # Overall uncertainty statistics - ensure JSON serializable
        uncertainty_stats = {
            'mean_uncertainty': float(np.mean(all_uncertainties)),
            'std_uncertainty': float(np.std(all_uncertainties)),
            'min_uncertainty': float(np.min(all_uncertainties)),
            'max_uncertainty': float(np.max(all_uncertainties))
        }
        
        response_data = convert_to_json_serializable({
            'calibration_results': calibration_results,
            'uncertainty_statistics': uncertainty_stats,
            'total_samples': int(len(all_labels)),  # Ensure int
            'note': 'Model calibration helps assess prediction reliability'
        })
        
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Model calibration error: {e}")
        return jsonify({'error': f'Calibration analysis failed: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    if not supabase:
        return jsonify({'error': 'Database unavailable'}), 500
    
    try:
        response = supabase.table('predictions').select('*').order('created_at', desc=True).execute()
        history = []
        
        for record in response.data:
            # Handle timestamp formatting
            created_at = record.get('created_at')
            if created_at:
                try:
                    if 'T' in created_at:
                        timestamp = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        timestamp = datetime.datetime.fromisoformat(created_at)
                    formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                except Exception as ts_error:
                    logger.warning(f"Timestamp parsing error: {ts_error}")
                    formatted_timestamp = str(created_at)
            else:
                formatted_timestamp = 'N/A'
            
            history_record = {
                'id': str(record['id']),
                'patientName': record.get('patient_name', 'N/A'),
                'patientAge': record.get('patient_age', 'N/A'),
                'patientGender': record.get('patient_gender', 'N/A'),
                'smokingHistory': record.get('smoking_history', 'N/A'),
                'analysisResults': record.get('analysis_results', []),
                'modelVersion': record.get('model_version', 'Enhanced Medical CNN'),
                'timestamp': formatted_timestamp
            }
            history.append(history_record)
        
        logger.info(f"Retrieved {len(history)} medical records from database")
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500

@app.route('/delete_history_item/<item_id>', methods=['DELETE'])
def delete_history_item(item_id):
    if not supabase:
        return jsonify({'error': 'Database unavailable'}), 500
    
    try:
        record_id = int(item_id)
        response = supabase.table('predictions').delete().eq('id', record_id).execute()
        
        if response.data:
            logger.info(f"Successfully deleted medical record with ID: {record_id}")
            return jsonify({'success': True, 'message': 'Record deleted successfully'})
        else:
            return jsonify({'error': 'Record not found or already deleted'}), 404
            
    except ValueError:
        return jsonify({'error': 'Invalid record ID'}), 400
    except Exception as e:
        logger.error(f"Failed to delete record {item_id}: {e}")
        return jsonify({'error': f'Failed to delete record: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

# --------------- Run App -----------------------
if __name__ == '__main__':
    logger.info("Starting Flask application with Enhanced Medical CNN Model")
    load_model_at_startup()
    
    # Log configuration status
    logger.info(f"Enhanced medical model loaded: {model is not None}")
    logger.info(f"Supabase connected: {supabase is not None}")
    logger.info(f"Device: {device}")
    logger.info(f"Class names: {class_names}")
    
    app.run(debug=True, host='0.0.0.0', port=5008)