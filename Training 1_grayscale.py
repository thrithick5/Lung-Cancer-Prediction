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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from typing import List, Tuple

# ========================================
# UTILITY FUNCTIONS
# ========================================

def clean_ds_store(root):
    """Remove all .DS_Store files recursively to prevent data loading issues on macOS."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == '.DS_Store':
                full_path = os.path.join(dirpath, name)
                try:
                    os.remove(full_path)
                    print(f"Removed: {full_path}")
                except OSError as e:
                    print(f"Error removing {full_path}: {e}")

# ========================================
# ADVANCED MEDICAL CNN FOR GRAYSCALE
# ========================================

class AdvancedMedicalCNN(nn.Module):
    """
    Advanced CNN architecture optimized for grayscale medical imaging.
    Features:
    - Residual connections for better gradient flow
    - Attention mechanisms for focusing on important regions
    - Designed specifically for 224x224 grayscale images
    """
    def __init__(self, num_classes, dropout_rate=0.4):
        super().__init__()
        self.dropout_rate = dropout_rate
        # Initial convolution - from 1 channel (grayscale) to 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Residual blocks
        self.res_block1 = self._make_residual_block(64, 128, stride=1)
        self.res_block2 = self._make_residual_block(128, 256, stride=2)
        self.res_block3 = self._make_residual_block(256, 512, stride=2)
        self.res_block4 = self._make_residual_block(512, 512, stride=2)
        # Spatial attention module
        self.spatial_attention = SpatialAttentionModule()
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _make_residual_block(self, in_channels, out_channels, stride):
        """Create a residual block with skip connection."""
        return ResidualBlock(in_channels, out_channels, stride)

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
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
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SpatialAttentionModule(nn.Module):
    """Spatial attention to focus on important image regions."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        return x * attention


# ========================================
# CALIBRATION MODULES
# ========================================

class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        """Tune temperature using validation set."""
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(eval_loss)
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        return self

# ========================================
# FOCAL LOSS FOR IMBALANCED DATA
# ========================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in medical data."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# ========================================
# TRAINING UTILITIES
# ========================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup data augmentation."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========================================
# EVALUATION & PREDICTION FUNCTIONS
# ========================================

def evaluate_model(model, data_loader, criterion, device):
    """Comprehensive model evaluation on a dataset."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
    }

def predict_single_image(model, image_path, class_names, device, transforms):
    """Predict the class and confidence for a single image."""
    model.eval()
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    image = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        confidence_percent = int(round(confidence.item() * 100))
        predicted_class = class_names[predicted_idx.item()]
    return predicted_class, confidence_percent

def predict_batch_images(model, image_paths: List[str], class_names: List[str], device, transforms) -> List[Tuple[str, str, int]]:
    """
    Predict the class and confidence for a batch of images simultaneously.
    Args:
        model: The trained PyTorch model.
        image_paths: A list of file paths for the images.
        class_names: A list of the class names.
        device: The device to run inference on ('cuda' or 'cpu').
        transforms: The transformations to apply to the images.
    Returns:
        A list of tuples, where each tuple contains (image_path, predicted_class, confidence_percent).
    """
    model.eval()
    batch_images = []
    valid_paths = []
    # Load and transform each image, skipping any that can't be opened
    for path in image_paths:
        try:
            image = Image.open(path)
            batch_images.append(transforms(image))
            valid_paths.append(path)
        except FileNotFoundError:
            print(f"Warning: Image file not found at {path}. Skipping.")
        except Exception as e:
            print(f"Warning: Could not open {path} due to {e}. Skipping.")

    if not batch_images:
        print("No valid images found to predict.")
        return []

    # Stack images into a single batch tensor
    batch_tensor = torch.stack(batch_images).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidences, predicted_indices = torch.max(probabilities, 1)

    # Compile results
    results = []
    for i in range(len(valid_paths)):
        path = valid_paths[i]
        confidence_percent = int(round(confidences[i].item() * 100))
        predicted_class = class_names[predicted_indices[i].item()]
        results.append((path, predicted_class, confidence_percent))

    return results


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path='training_history.png'):
    """Plot training and validation history for loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history saved: {save_path}")


def plot_confusion_matrix(labels, predictions, class_names, save_path='confusion_matrix.png'):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved: {save_path}")


# ========================================
# MAIN TRAINING SCRIPT
# ========================================

if __name__ == '__main__':
    print("="*70)
    print("GRAYSCALE MEDICAL IMAGE TRAINING - ADVANCED CNN")
    print("="*70)
    train_dir, val_dir = 'processed_dataset/train', 'processed_dataset/val'
    clean_ds_store(train_dir)
    clean_ds_store(val_dir)

    # ---- HYPERPARAMETERS ----
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ---- DATA TRANSFORMS ----
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ---- LOAD DATASETS ----
    print("\nLoading datasets...")
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        if not train_dataset or not val_dataset: raise ValueError("Dataset is empty.")
        # The fix is here: added drop_last=True to the train_loader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        class_names = train_dataset.classes
        num_classes = len(class_names)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] Failed to load datasets: {e}")
        exit()

    print(f"Classes: {class_names}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # ---- MODEL, LOSS, OPTIMIZER ----
    model = AdvancedMedicalCNN(num_classes=num_classes).to(DEVICE)
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=15)

    # ---- TRAINING LOOP ----
    print("\n" + "="*70 + "\nSTARTING TRAINING\n" + "="*70)
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if USE_MIXUP:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, DEVICE)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (lam * predicted.eq(labels_a.data).sum().item() + (1 - lam) * predicted.eq(labels_b.data).sum().item()) if USE_MIXUP else (predicted == labels).sum().item()

        # ---- VALIDATION ----
        val_results = evaluate_model(model, val_loader, nn.CrossEntropyLoss(), DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss / len(train_loader.dataset))
        history['train_acc'].append(train_correct / train_total)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])

        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")

        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, 'best_model.pth')
            print(f"  -> New best model saved with accuracy: {best_val_acc:.4f}")
        
        early_stopping(val_results['loss'])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nTRAINING COMPLETED\n" + "="*70)

    # ---- FINAL EVALUATION & VISUALIZATION ----
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_results = evaluate_model(model, val_loader, nn.CrossEntropyLoss(), DEVICE)
    print(f"\nFinal Validation Accuracy: {final_results['accuracy']:.4f}")
    print("\nClassification Report:\n", classification_report(final_results['labels'], final_results['predictions'], target_names=class_names, digits=4))
    plot_training_history(history['train_loss'], history['val_loss'], history['train_acc'], history['val_acc'])
    plot_confusion_matrix(final_results['labels'], final_results['predictions'], class_names)

    # ---- INFERENCE EXAMPLES ----
    print("\n" + "="*70 + "\nINFERENCE EXAMPLES\n" + "="*70)
    try:
        # --- 1. Single Image Inference ---
        print("--- Single Image Prediction ---")
        sample_img_path, _ = val_dataset.samples[0]
        predicted_class, confidence = predict_single_image(model, sample_img_path, class_names, DEVICE, val_transforms)
        if predicted_class is not None:
            print(f"Image: '{os.path.basename(sample_img_path)}'")
            print(f"-> Predicted Class: {predicted_class}, Confidence: {confidence}%")

        # --- 2. Batch Image Inference ---
        print("\n--- Batch Image Prediction (processing 4 scans simultaneously) ---")
        num_batch_samples = min(4, len(val_dataset.samples))
        if num_batch_samples > 1:
            batch_image_paths = [val_dataset.samples[i][0] for i in range(num_batch_samples)]
            batch_results = predict_batch_images(model, batch_image_paths, class_names, DEVICE, val_transforms)
            for path, pred_class, conf in batch_results:
                print(f"Image: '{os.path.basename(path)}'")
                print(f"-> Predicted Class: {pred_class}, Confidence: {conf}%")
        else:
            print("Not enough validation images to run a batch prediction example.")

    except IndexError:
        print("\nCould not run inference example: Validation dataset is empty.")
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")

    print("\n" + "="*70 + "\nPIPELINE COMPLETED SUCCESSFULLY\n" + "="*70)


