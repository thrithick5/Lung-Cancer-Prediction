import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle

# ---- Step 1: Clean all .DS_Store files ----
def clean_ds_store(root):
    """Remove all .DS_Store files recursively"""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == '.DS_Store':
                full_path = os.path.join(dirpath, name)
                try:
                    os.remove(full_path)
                    print(f"Removed .DS_Store: {full_path}")
                except OSError as e:
                    print(f"Error removing {full_path}: {e}")

# ---- Step 2: Filter out non-folder items from class detection ----
def get_valid_class_folders(directory):
    """Get only valid class folders (ignore .DS_Store and other files)"""
    items = os.listdir(directory)
    valid_folders = []
    for item in items:
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            valid_folders.append(item)
    return sorted(valid_folders)

# ---- Enhanced Medical CNN Model with Uncertainty Estimation ----
class EnhancedMedicalCNN(nn.Module):
    """Enhanced Medical CNN with Monte Carlo Dropout for uncertainty estimation"""
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
        self.feature_size = 256 * (224 // 16) * (224 // 16)
        
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
        
        self.eval()
        return torch.stack(predictions)

# ---- Temperature Scaling for Calibration ----
class TemperatureScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification model
    """
    def __init__(self, model):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, device):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).to(device)
            labels = torch.cat(labels_list).to(device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI. 2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

# ---- Medical Interpretation Functions ----
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
        'predictive_entropy': entropy.item(),
        'max_confidence': max_confidence.item()
    }

def interpret_medical_prediction_raw(probabilities, std_probabilities, entropy, class_names, confidence_threshold=0.7):
    """
    Provide raw medical interpretation without artificial mapping
    Reports actual model confidence and uncertainty
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
    risk_assessment = categorize_medical_risk_truthful(predicted_class, confidence, reliability)
    
    # Prepare detailed class probabilities with uncertainties
    detailed_probabilities = []
    for i, class_name in enumerate(class_names):
        detailed_probabilities.append({
            'class': class_name,
            'raw_probability': round(prob_percentages[i], 2),
            'uncertainty': round(std_percentages[i], 2),
            'confidence_interval_95': f"{max(0, prob_percentages[i] - 1.96*std_percentages[i]):.1f}% - {min(100, prob_percentages[i] + 1.96*std_percentages[i]):.1f}%"
        })
    
    return {
        'predicted_class': predicted_class,
        'model_confidence': round(confidence, 2),
        'prediction_uncertainty': round(uncertainty, 2),
        'predictive_entropy': round(entropy, 3),
        'reliability': reliability,
        'reliability_note': reliability_note,
        'risk_assessment': risk_assessment,
        'detailed_probabilities': detailed_probabilities,
        'model_notes': f"Highest class probability: {confidence:.1f}% ± {uncertainty:.1f}%",
        'medical_disclaimer': "This AI analysis provides RAW model outputs and must be reviewed by qualified medical professionals. These are computational probabilities, not clinical diagnoses."
    }

def categorize_medical_risk_truthful(predicted_class, confidence, reliability):
    """Categorize medical risk truthfully based on actual model performance"""
    class_lower = predicted_class.lower()
    
    base_recommendation = "Clinical correlation and professional medical interpretation required for all AI-assisted findings."
    
    if class_lower == 'normal':
        if reliability == "High" and confidence > 80:
            return {
                'level': f'Model suggests normal findings (confidence: {confidence:.1f}%)',
                'recommendation': f'{base_recommendation} Consider routine follow-up as clinically indicated.',
                'urgency': 'Routine clinical workflow',
                'interpretation_note': 'High model confidence in normal classification'
            }
        else:
            return {
                'level': f'Model suggests normal findings with uncertainty (confidence: {confidence:.1f}%)',
                'recommendation': f'{base_recommendation} Consider additional evaluation if clinically indicated.',
                'urgency': 'Clinical correlation recommended',
                'interpretation_note': 'Model shows uncertainty - clinical judgment essential'
            }
    
    elif class_lower in ['benign', 'beging']:
        if reliability == "High" and confidence > 70:
            return {
                'level': f'Model suggests benign findings (confidence: {confidence:.1f}%)',
                'recommendation': f'{base_recommendation} Follow standard clinical protocols for benign findings.',
                'urgency': 'Standard clinical workflow',
                'interpretation_note': 'Model indicates likely benign pathology - confirm with radiology review'
            }
        else:
            return {
                'level': f'Model suggests possible benign findings with uncertainty (confidence: {confidence:.1f}%)',
                'recommendation': f'{base_recommendation} Additional imaging or specialist consultation may be warranted.',
                'urgency': 'Enhanced clinical review recommended',
                'interpretation_note': 'Model uncertainty present - thorough clinical evaluation needed'
            }
    
    elif class_lower == 'malignant':
        return {
            'level': f'Model suggests concerning findings (confidence: {confidence:.1f}%)',
            'recommendation': f'{base_recommendation} Urgent specialist review and appropriate diagnostic workup indicated.',
            'urgency': 'Priority clinical attention - do not delay',
            'interpretation_note': 'Model suggests malignant features - immediate professional interpretation required'
        }
    
    else:
        return {
            'level': f'Unclear model output (confidence: {confidence:.1f}%)',
            'recommendation': f'{base_recommendation} Manual review of AI output and clinical re-evaluation.',
            'urgency': 'Clinical review required',
            'interpretation_note': 'Model output requires interpretation'
        }

# ---- Enhanced Training Function ----
def evaluate_model_calibration(model, valid_loader, device, class_names):
    """Evaluate model calibration using reliability plots and calibration metrics"""
    model.eval()
    all_probs = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for imgs, labels in valid_loader:
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
    
    # Calculate Expected Calibration Error
    ece_criterion = ECELoss()
    logits = torch.from_numpy(np.log(all_probs + 1e-8))
    labels_tensor = torch.from_numpy(all_labels)
    ece = ece_criterion(logits, labels_tensor)
    
    print(f"Expected Calibration Error: {ece.item():.4f}")
    
    # Calculate Brier Score for each class
    brier_scores = []
    for i, class_name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        brier_score = brier_score_loss(binary_labels, class_probs)
        brier_scores.append(brier_score)
        print(f"Brier Score for {class_name}: {brier_score:.4f}")
    
    return ece.item(), brier_scores, all_probs, all_labels, all_uncertainties

def predict_ct_scan_with_uncertainty(model, image_tensor, class_names, device):
    """Make prediction with proper uncertainty quantification"""
    try:
        # Calculate uncertainty
        uncertainty_results = calculate_prediction_uncertainty(model, image_tensor, device)
        
        # Get raw medical interpretation
        result = interpret_medical_prediction_raw(
            uncertainty_results['mean_probabilities'],
            uncertainty_results['std_probabilities'],
            uncertainty_results['predictive_entropy'],
            class_names
        )
        
        return result
    
    except Exception as e:
        print(f"Error in uncertainty prediction: {e}")
        raise

if __name__ == '__main__':
    # Clean .DS_Store files first
    clean_ds_store('processed_dataset/train')
    clean_ds_store('processed_dataset/val')

    # ---- Parameters ----
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 30  # Increased for better calibration
    NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ---- Paths ----
    train_dir = 'processed_dataset/train'
    val_dir = 'processed_dataset/val'

    # ---- Check if directories exist ----
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    # ---- Enhanced Transforms ----
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),  # Reduced rotation for medical images
        transforms.RandomHorizontalFlip(p=0.3),  # Reduced flip probability
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Gentle color augmentation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- Check folders ----
    train_classes = get_valid_class_folders(train_dir)
    val_classes = get_valid_class_folders(val_dir)

    print("Train class folders:", train_classes)
    print("Val class folders:", val_classes)

    if set(train_classes) != set(val_classes):
        print("WARNING: Train and validation folders have different classes!")
        print(f"Train only: {set(train_classes) - set(val_classes)}")
        print(f"Val only: {set(val_classes) - set(train_classes)}")

    # ---- Image file filtering ----
    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    def is_image_file(filename):
        return filename.lower().endswith(IMG_EXTS)

    # ---- Datasets & Dataloaders ----
    try:
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf, is_valid_file=is_image_file)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf, is_valid_file=is_image_file)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Detected classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")

    # ---- Enhanced Medical Model ----
    model = EnhancedMedicalCNN(num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Medical model created with {total_params:,} total parameters ({trainable_params:,} trainable)")

    # ---- Loss, Optimizer, Scheduler with medical-appropriate weighting ----
    # Calculate class weights for balanced training
    class_counts = {}
    for _, class_idx in train_ds.samples:
        class_name = train_ds.classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (len(class_counts) * class_counts[class_name]) 
                    for class_name in train_ds.classes]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} samples")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Use focal loss for medical imbalanced data
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)  # Reduced label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ---- Training Loop with Enhanced Medical Metrics ----
    print(f"\nStarting medical AI training for {EPOCHS} epochs...")
    best_val_acc = 0.0
    best_calibration_score = float('inf')
    patience = 10  # Increased patience for medical models
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            preds = outputs.argmax(dim=1)
            train_loss += loss.item() * imgs.size(0)
            train_correct += (preds == labels).sum().item()

        epoch_loss = train_loss / len(train_ds)
        epoch_acc = train_correct / len(train_ds)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Update learning rate
        scheduler.step()
        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s | "
              f"LR: {current_lr:.2e}")

        # Enhanced model saving with calibration consideration
        save_model = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_model = True
            print(f"✓ New best validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1

        # Evaluate calibration every 5 epochs
        if epoch % 5 == 0:
            print("Evaluating model calibration...")
            ece, brier_scores, _, _, _ = evaluate_model_calibration(model, val_loader, DEVICE, class_names)
            if ece < best_calibration_score:
                best_calibration_score = ece
                save_model = True
                print(f"✓ New best calibration score: {ece:.4f}")

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_calibration_score': best_calibration_score,
                'class_names': class_names,
                'class_weights': class_weights.cpu(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
                'model_type': 'Enhanced Medical CNN with Uncertainty'
            }, "best_lung_cancer_cnn_enhanced.pth")
            print(f"Enhanced medical model saved!")
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break

    print(f"\nMedical AI training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Calibration Score (ECE): {best_calibration_score:.4f}")
    print(f"Model saved as: best_lung_cancer_cnn_enhanced.pth")
    
    # ---- Post-training Calibration ----
    print("\n" + "="*60)
    print("POST-TRAINING MODEL CALIBRATION")
    print("="*60)
    
    # Load the best model
    checkpoint = torch.load("best_lung_cancer_cnn_enhanced.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply temperature scaling
    print("Applying temperature scaling for better calibration...")
    calibrated_model = TemperatureScaling(model)
    calibrated_model.set_temperature(val_loader, DEVICE)
    
    # Save calibrated model
    torch.save({
        'base_model_state_dict': model.state_dict(),
        'temperature': calibrated_model.temperature.item(),
        'class_names': class_names,
        'model_type': 'Temperature-Calibrated Medical CNN',
        'training_info': checkpoint
    }, "calibrated_medical_cnn.pth")
    
    print("Temperature-calibrated model saved as: calibrated_medical_cnn.pth")
    
    # ---- Final Model Evaluation with Raw Probabilities ----
    print("\n" + "="*60)
    print("MEDICAL AI EVALUATION - RAW PROBABILITY ANALYSIS")
    print("="*60)
    
    # Evaluate final calibration
    final_ece, final_brier, all_probs, all_labels, all_uncertainties = evaluate_model_calibration(model, val_loader, DEVICE, class_names)
    
    # Demo with raw probability interpretation
    model.eval()
    with torch.no_grad():
        # Get a random sample from validation set
        sample_idx = torch.randint(0, len(val_ds), (1,)).item()
        sample_image, sample_label = val_ds[sample_idx]
        
        # Get raw medical prediction
        result = predict_ct_scan_with_uncertainty(model, sample_image, class_names, DEVICE)
        
        print(f"\nSample Medical AI Analysis:")
        print(f"True Label: {class_names[sample_label]}")
        print(f"AI Prediction: {result['predicted_class']}")
        print(f"Model Confidence: {result['model_confidence']:.1f}%")
        print(f"Prediction Uncertainty: ±{result['prediction_uncertainty']:.1f}%")
        print(f"Reliability Assessment: {result['reliability']}")
        print(f"Risk Assessment: {result['risk_assessment']['level']}")
        print(f"Clinical Recommendation: {result['risk_assessment']['recommendation']}")
        print(f"\nDetailed Class Probabilities:")
        for prob_info in result['detailed_probabilities']:
            print(f"  {prob_info['class']}: {prob_info['raw_probability']:.1f}% ± {prob_info['uncertainty']:.1f}% (CI: {prob_info['confidence_interval_95']})")
        print(f"\nModel Notes: {result['model_notes']}")
        print(f"Medical Disclaimer: {result['medical_disclaimer']}")
    
    # Generate calibration plots
    plt.figure(figsize=(15, 10))
    
    # Reliability diagram
    plt.subplot(2, 3, 1)
    for i, class_name in enumerate(class_names):
        binary_labels = (all_labels == i).astype(int)
        class_probs = all_probs[:, i]
        
        # Create bins and calculate calibration
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
            if in_bin.sum() > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(binary_labels[in_bin].mean())
        
        plt.plot(bin_centers, bin_accuracies, 'o-', label=f'{class_name}', alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty histogram
    plt.subplot(2, 3, 2)
    plt.hist(all_uncertainties, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    plt.grid(True, alpha=0.3)
    
    # Confidence vs Accuracy scatter
    plt.subplot(2, 3, 3)
    max_probs = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    correct = (predictions == all_labels).astype(int)
    
    plt.scatter(max_probs, correct, alpha=0.5, s=20)
    plt.xlabel('Max Probability (Confidence)')
    plt.ylabel('Correct Prediction')
    plt.title('Confidence vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Class-wise performance
    plt.subplot(2, 3, 4)
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = (all_labels == i)
        if class_mask.sum() > 0:
            class_pred = np.argmax(all_probs[class_mask], axis=1)
            class_acc = (class_pred == i).mean()
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.bar(range(len(class_names)), class_accuracies, alpha=0.7)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Brier score comparison
    plt.subplot(2, 3, 5)
    plt.bar(range(len(class_names)), final_brier, alpha=0.7)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylabel('Brier Score')
    plt.title('Brier Score by Class (Lower is Better)')
    plt.grid(True, alpha=0.3)
    
    # ECE visualization
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.6, f'Expected Calibration Error\n{final_ece:.4f}', 
             ha='center', va='center', fontsize=14, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.text(0.5, 0.4, f'Mean Brier Score\n{np.mean(final_brier):.4f}', 
             ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Overall Calibration Metrics')
    
    plt.tight_layout()
    plt.savefig('medical_model_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCalibration analysis saved as: medical_model_calibration_analysis.png")
    
    # ---- Medical AI Usage Guidelines ----
    print("\n" + "="*60)
    print("MEDICAL AI USAGE GUIDELINES")
    print("="*60)
    print("""
IMPORTANT MEDICAL AI GUIDELINES:

1. RAW PROBABILITY REPORTING:
   - This model now reports actual model confidence without artificial mapping
   - Probabilities reflect the model's genuine uncertainty
   - Higher uncertainty indicates areas requiring careful review

2. UNCERTAINTY QUANTIFICATION:
   - Monte Carlo Dropout provides uncertainty estimates
   - High uncertainty (>10%) suggests model is unsure
   - Low confidence (<70%) requires additional clinical evaluation

3. CALIBRATION AWARENESS:
   - Expected Calibration Error (ECE): {:.4f}
   - Lower ECE indicates better-calibrated probabilities
   - Temperature scaling applied for improved calibration

4. CLINICAL INTEGRATION:
   - ALL predictions require professional medical review
   - Use as a screening/triage tool, not for diagnosis
   - Pay attention to uncertainty measures and reliability scores
   - Consider clinical context and patient history

5. APPROPRIATE USE CASES:
   - Initial screening and flagging concerning cases
   - Quality assurance and second opinion support
   - Research and educational applications
   - Workload prioritization in clinical settings

6. LIMITATIONS:
   - Model trained on specific dataset - may not generalize
   - Cannot replace radiologist interpretation
   - Uncertainty estimates are approximate
   - Performance may vary across different populations

7. RECOMMENDED WORKFLOW:
   - Use AI for initial assessment
   - Review uncertainty and reliability scores
   - Integrate with clinical findings
   - Confirm with qualified medical professionals
   - Document AI assistance in clinical notes

Remember: This AI system provides computational assessments, not medical diagnoses.
All clinical decisions must involve qualified healthcare professionals.
""".format(final_ece))
    
    print("Enhanced medical AI training complete with proper uncertainty quantification!")
    print("Model files:")
    print("- best_lung_cancer_cnn_enhanced.pth (base model)")
    print("- calibrated_medical_cnn.pth (temperature-calibrated model)")
    print("- medical_model_calibration_analysis.png (calibration plots)")