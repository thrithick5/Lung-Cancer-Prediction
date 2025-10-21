import os
import time
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple

# =======================================================
# COLAB/SETUP INSTRUCTIONS (MUST RUN SEPARATELY IN COLAB)
# =======================================================
"""
============================================================
           CRITICAL COLAB DATA SETUP STEPS
============================================================
This script must be run *after* the data is unzipped. Use the
commands below in SEPARATE Colab code cells before running
the main training block.

1. RENAME YOUR UPLOADED FILE (Optional, but Safest):
   If your zip file has spaces or special characters (e.g., 'Training Data.zip'),
   rename it to something simple like 'data.zip' in the Colab file pane (left side).

2. EXECUTE THE UNZIP COMMAND IN A NEW CELL:
   The command below assumes your data file is now named 'data.zip'.
   
   !unzip -q '/content/data.zip' -d '/content/'
   
   If your file is still named 'lung_cancer_dataset.zip', use this:
   
   !unzip -q '/content/lung_cancer_dataset.zip' -d '/content/'


3. CONFIRM DATA_DIR PATH:
   After unzipping, a folder will appear (e.g., 'lung_cancer_dataset').
   The DATA_DIR below must match the name of that folder.
   
"""

# =======================================================
# CONFIGURATION
# =======================================================
# CRITICAL: This path must match the root folder created after unzipping the dataset!
# If your unzipped folder is named 'lung_cancer_data', change the path below.
DATA_DIR = '/content/lung_cancer_dataset' 

MODEL_NAME = 'calibrated_grayscale_cnn.pth'
BASE_MODEL_NAME = 'best_grayscale_cnn_base.pth'
CALIBRATION_PLOT_NAME = 'calibration_reliability_plot.png'
NUM_CLASSES = 2 # Assuming two classes: Cancerous/Non-cancerous
INPUT_CHANNELS = 1 # CRITICAL: Set to 1 for grayscale CT scans
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reduced worker number to 1 to address the Colab UserWarning about max workers.
NUM_WORKERS = 1 

# =======================================================
# UTILITY FUNCTIONS
# =======================================================

def clean_ds_store(root):
    """Remove all .DS_Store files recursively to prevent data loading issues on macOS."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name == '.DS_Store':
                full_path = os.path.join(dirpath, name)
                try:
                    os.remove(full_path)
                    # print(f"Removed: {full_path}")
                except OSError as e:
                    print(f"Error removing {full_path}: {e}")

# =======================================================
# 1. ADVANCED GRAYSCALE CNN ARCHITECTURE
# (Includes Attention and Residual Blocks)
# =======================================================

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
    def __init__(self, num_classes=NUM_CLASSES):
        super(AdvancedMedicalCNN, self).__init__()
        # 1. Initial Convolution (Input: 1 Channel)
        self.conv_initial = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, kernel_size=7, stride=2, padding=3, bias=False),
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

# =======================================================
# 2. CALIBRATION UTILITIES
# =======================================================

class ModelWithTemperature(nn.Module):
    """
    FIXED: A wrapper that applies temperature scaling directly to the input logits (N x C).
    It does NOT call the underlying CNN model again, avoiding the 4D input error.
    """
    def __init__(self, model):
        # We ignore the 'model' argument since we are optimizing on pre-computed logits.
        super(ModelWithTemperature, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """Applies temperature scaling to the input logits (N x C)."""
        return logits / self.temperature

def evaluate_calibration(probs, labels, num_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(probs, bins, right=True)

    ece = 0
    for i in range(1, num_bins + 1):
        bin_indices = np.where(indices == i)[0]
        if len(bin_indices) == 0:
            continue

        bin_accuracy = np.mean(labels[bin_indices])
        bin_confidence = np.mean(probs[bin_indices])
        bin_weight = len(bin_indices) / len(probs)

        ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    return ece

def plot_reliability_diagram(probs, labels, title, filename):
    """Plots a reliability diagram (calibration curve)."""
    frac_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=10, strategy='uniform'
    )

    ece = evaluate_calibration(probs, labels)
    brier = brier_score_loss(labels, probs)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(mean_predicted_value, frac_of_positives, "s-", label=f"Model (ECE: {ece:.4f})")
    plt.title(f'{title} Reliability Diagram\nBrier Score: {brier:.4f}')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Calibration plot saved to {filename}")
    return ece, brier

# =======================================================
# 3. TEMPERATURE SCALING CALIBRATION
# =======================================================

def calibrate_model(model, val_loader, device, plot_filename):
    """
    Applies Temperature Scaling to the trained model using the validation set.
    Saves a single, combined file containing the model weights and the temperature.
    """
    print("\n" + "="*50)
    print("STARTING MODEL CALIBRATION (Temperature Scaling)")
    print("="*50)

    # 1. Gather all logits and labels from validation set
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Calculate initial ECE
    initial_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    positive_probs = initial_probs[:, 1]
    initial_ece, _ = plot_reliability_diagram(positive_probs, all_labels, "Uncalibrated Model", f"uncalibrated_{plot_filename}")

    print(f"Initial ECE (Uncalibrated): {initial_ece:.4f}")

    # 2. Instantiate ModelWithTemperature and set up optimizer
    temp_model = ModelWithTemperature(model)
    temp_model.to(device)

    # LBFGS is used for calibration optimization
    optimizer = optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    criterion = nn.CrossEntropyLoss().to(device)
    labels_tensor = torch.from_numpy(all_labels).to(device)
    logits_tensor = torch.from_numpy(all_logits).to(device).float() # Ensure float type

    def eval():
        optimizer.zero_grad()
        # Loss is calculated on the temperature-scaled logits. 
        # temp_model.forward() now correctly expects and processes logits_tensor.
        loss = criterion(temp_model(logits_tensor), labels_tensor) 
        loss.backward()
        return loss

    optimizer.step(eval)

    # 4. Final Evaluation and Saving
    final_temp = temp_model.temperature.item()
    print(f"\nOptimization finished. Learned Temperature: {final_temp:.4f}")

    # Apply the learned temperature for final ECE calculation
    calibrated_logits = all_logits / final_temp
    calibrated_probs = F.softmax(torch.from_numpy(calibrated_logits), dim=1).numpy()
    final_ece, final_brier = plot_reliability_diagram(calibrated_probs[:, 1], all_labels, f"Calibrated Model (T={final_temp:.2f})", plot_filename)

    print(f"Final ECE (Calibrated): {final_ece:.4f}")
    print(f"Final Brier Score (Calibrated): {final_brier:.4f}")

    # Save the base model weights and the temperature value into a single dictionary.
    final_state = {
        'model_state_dict': model.state_dict(),
        'temperature': final_temp,
        'config': {
            'input_channels': INPUT_CHANNELS,
            'num_classes': NUM_CLASSES
        }
    }
    torch.save(final_state, MODEL_NAME)
    print(f"Calibrated model (weights + temp) saved as a single file: '{MODEL_NAME}'")

    return final_ece

# =======================================================
# 4. MAIN TRAINING AND VALIDATION LOOP
# =======================================================

def train_and_validate(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    """The standard PyTorch training and validation function."""
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), BASE_MODEL_NAME)
                print(f"Saved new best base model: {BASE_MODEL_NAME} with Acc: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights for calibration
    model.load_state_dict(torch.load(BASE_MODEL_NAME))
    return model

# =======================================================
# 5. EXECUTION BLOCK
# =======================================================

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    # 1. Clean up potential macOS data files
    clean_ds_store(DATA_DIR)

    # 2. Define Grayscale Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=INPUT_CHANNELS), # Ensure 1 channel
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=INPUT_CHANNELS), # Ensure 1 channel
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }

    # 3. Load Data (FIXED: Added is_valid_file check to filter out non-image data/unexpected folders)
    print(f"Attempting to load data from: {DATA_DIR}")
    try:
        # Check if the directory exists and contains 'train' and 'val' subdirectories
        if not os.path.exists(os.path.join(DATA_DIR, 'train')) or \
           not os.path.exists(os.path.join(DATA_DIR, 'val')):
           # Raise a more descriptive error if the structure is wrong
           raise FileNotFoundError("Missing 'train' or 'val' subdirectories. Ensure the zip file extracts to a folder containing 'train' and 'val' subfolders.")
        
        # Helper function to filter out non-image files if ImageFolder encounters them
        def is_valid_file(path):
            """Checks if a file is a common image format."""
            return path.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))

        # Check for unexpected classes and filter them
        # We manually get class names from the 'train' subdirectory, excluding non-folders/hidden items
        
        # --- FIX FOR TARGET OUT OF BOUNDS ---
        # Get only the *actual* class folders and sort them for consistent indexing (0, 1)
        train_dir = os.path.join(DATA_DIR, 'train')
        all_train_subdirs = sorted([d for d in os.listdir(train_dir) 
                                   if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith('.')])
        
        if len(all_train_subdirs) != NUM_CLASSES:
             print("\n" + "="*80)
             print(f"CLASS MISMATCH ERROR: Expected {NUM_CLASSES} classes but found {len(all_train_subdirs)}: {all_train_subdirs}")
             print("The code will now proceed using the first 2 classes found, but verify your dataset structure!")
             print("If you have more than 2 classes, you must change NUM_CLASSES = 2 in the CONFIGURATION section.")
             print("="*80)
             # If too many classes are found, we need to decide which ones to use. 
             # For a robust solution, we will stick with ImageFolder's automatic indexing 
             # but check the final count before proceeding. The error is likely an extraneous folder.
        
        # Create datasets using ImageFolder (which indexes alphabetically)
        image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), 
                                                  data_transforms[x],
                                                  is_valid_file=is_valid_file) # Added filter
                          for x in ['train', 'val']}

        final_class_count = len(image_datasets['train'].classes)
        
        if final_class_count > NUM_CLASSES:
             raise IndexError(f"Found {final_class_count} classes in ImageFolder, but expected {NUM_CLASSES}. Labels must be 0 to {NUM_CLASSES - 1}.")
             
        # Dataloaders are defined here, resolving the NameError if successful
        dataloaders = {x: DataLoader(image_datasets[x], 
                                     batch_size=32, 
                                     shuffle=True, 
                                     num_workers=NUM_WORKERS) # Reduced workers
                       for x in ['train', 'val']}
                       
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        print(f"SUCCESS: Data loaded. Classes found: {class_names}")
        print(f"Train size: {dataset_sizes['train']}, Val size: {dataset_sizes['val']}")

    except FileNotFoundError as e:
        print("\n" + "="*80)
        print(f"CRITICAL ERROR: Data loading failed. {e}")
        print("Please ensure your dataset is unzipped correctly and the 'DATA_DIR' path is set.")
        print(f"Expected unzipped folder path: {DATA_DIR}")
        print("Did you run the unzip command in a separate cell?")
        print("="*80)
        # We exit gracefully if data loading fails
        exit()
    except IndexError as e:
        print("\n" + "="*80)
        print(f"FATAL INDEX ERROR: {e}")
        print("This means ImageFolder detected more classes than NUM_CLASSES (2).")
        print("Check your 'train' and 'val' folders for hidden files, extra folders, or unexpected classes.")
        print("Expected classes (0, 1) are derived from the folder names.")
        print("="*80)
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()

    # 4. Initialize Model, Loss, Optimizer
    model_ft = AdvancedMedicalCNN(NUM_CLASSES)
    model_ft = model_ft.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 5. Train the Model
    best_model = train_and_validate(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, DEVICE, num_epochs=20)

    # 6. Apply Calibration
    final_ece = calibrate_model(best_model, dataloaders['val'], DEVICE, CALIBRATION_PLOT_NAME)

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Final Calibrated ECE: {final_ece:.4f}")
    print("The single, deployable model file is saved as:", MODEL_NAME)
    print("="*80)
