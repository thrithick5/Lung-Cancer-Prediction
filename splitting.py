import os
import shutil
import random
from pathlib import Path

def split_dataset(source_dir, dest_dir, train_split=0.8):
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Create train and val directories
    for split in ['train', 'val']:
        for class_dir in os.listdir(source_dir):
            class_path = source_dir / class_dir
            if class_path.is_dir():  # Ensure it's a directory
                os.makedirs(dest_dir / split / class_dir, exist_ok=True)

    # Split and copy
    for class_name in os.listdir(source_dir):
        class_path = source_dir / class_name
        if class_path.is_dir():  # Ensure it's a directory
            images = [img for img in os.listdir(class_path) if img.endswith(('.png', '.jpg', '.jpeg'))]  # Filter image files
            random.shuffle(images)

            split_idx = int(len(images) * train_split)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            for img in train_images:
                shutil.copy(class_path / img, dest_dir / 'train' / class_name / img)

            for img in val_images:
                shutil.copy(class_path / img, dest_dir / 'val' / class_name / img)

# Example usage
split_dataset('resized_dataset', 'processed_dataset')