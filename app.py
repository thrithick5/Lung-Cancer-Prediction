import os
import logging
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from torchvision import transforms
from dotenv import load_dotenv
import psycopg2

# ========================================
# PATH SETUP
# ========================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "calibrated_grayscale_cnn.pth"
UPLOAD_FOLDER_PATH = APP_DIR / "uploads"

# ========================================
# ENVIRONMENT VARIABLES
# ========================================
env_path = APP_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

DATABASE_URL = os.getenv("DATABASE_URL")

# ========================================
# LOGGING
# ========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# FLASK SETUP
# ========================================
app = Flask(__name__)
CORS(app)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER_PATH)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ========================================
# DATABASE CONNECTION (NEON)
# ========================================
def get_db_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in .env")
    return psycopg2.connect(DATABASE_URL)


# ========================================
# MODEL ARCHITECTURE
# ========================================
class AttentionModule(nn.Module):
    """Simple Spatial and Channel Attention Block."""
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
    """A standard Residual Block."""
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
    """Advanced CNN architecture optimized for grayscale medical imaging (1 Channel)."""
    def __init__(self, num_classes=2):
        super(AdvancedMedicalCNN, self).__init__()
        # 1. Initial Convolution (Input: 1 Channel)
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
        """Helper function to stack residual blocks."""
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


# ========================================
# MODEL LOADING
# ========================================
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["Beginning", "Malignant"]
model_temperature = 1.0

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def load_model_at_startup():
    global model, model_temperature

    if not MODEL_PATH.exists():
        logger.error("Model file not found.")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model = AdvancedMedicalCNN(num_classes=len(class_names))

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

        if "temperature" in checkpoint:
            model_temperature = checkpoint["temperature"]
            logger.info(f"Loaded temperature scaling factor: {model_temperature}")

        model.to(device)
        model.eval()

        logger.info("Model loaded successfully.")

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        model = None


# ========================================
# HELPER FUNCTIONS
# ========================================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def predict_single_image(image_path):
    try:
        image = Image.open(image_path).convert("L")
        tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            scaled_outputs = outputs / model_temperature
            probs = F.softmax(scaled_outputs, dim=1)[0]
            confidence, idx = torch.max(probs, 0)

        return {
            "predicted_class_name": class_names[idx.item()],
            "confidence_percentage": int(round(confidence.item() * 100)),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {"error": "Failed to process image."}


# ========================================
# ROUTES
# ========================================
@app.route("/")
def home():
    return render_template("frontpage.html")


@app.route("/detector")
def detector():
    return render_template("index.html")


@app.route("/model_status")
def model_status():
    return jsonify({
        "loaded": model is not None,
        "device": str(device),
        "class_names": class_names,
    })


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded."}), 503

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded."}), 400

    patient_info = {
        "patientName": request.form.get("patientName", "Anonymous"),
        "patientAge": request.form.get("patientAge"),
        "patientGender": request.form.get("patientGender"),
        "smokingHistory": request.form.get("smokingHistory"),
    }

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result = predict_single_image(path)
            result["filename"] = file.filename
            results.append(result)

            os.remove(path)
        else:
            results.append({"filename": file.filename, "error": "Invalid file type."})

    # ==========================
    # SAVE TO NEON DATABASE
    # ==========================
    first_valid = next((r for r in results if "error" not in r), None)

    if first_valid:
        try:
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO public.predictions
                (patient_name, age, gender, smoking_history, predicted_class, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                patient_info["patientName"],
                int(patient_info["patientAge"]) if patient_info["patientAge"] else None,
                patient_info["patientGender"],
                patient_info["smokingHistory"],
                first_valid["predicted_class_name"],
                float(first_valid["confidence_percentage"]),
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info("Prediction saved to Neon.")

        except Exception as e:
            logger.error(f"Database insert failed: {e}", exc_info=True)

    return jsonify({
        "patient_info": patient_info,
        "results": results
    })

if __name__ == "__main__":
    load_model_at_startup()
    app.run(debug=True, host="0.0.0.0", port=5008)