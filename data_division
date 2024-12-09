import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
original_dataset_path = r"C:\Users\ASUS\PycharmProjects\BeyondWords\Final_Dataset"
train_dir = r"C:\Users\ASUS\PycharmProjects\BeyondWords\train_dataset"
val_dir = r"C:\Users\ASUS\PycharmProjects\BeyondWords\val_dataset"
test_dir = r"C:\Users\ASUS\PycharmProjects\BeyondWords\test_dataset"

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split data
for gesture in os.listdir(original_dataset_path):
    gesture_path = os.path.join(original_dataset_path, gesture)
    images = os.listdir(gesture_path)

    # Train-Val-Test split
    train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.3, random_state=42)

    # Move images to respective folders
    for img in train_images:
        shutil.copy(os.path.join(gesture_path, img), os.path.join(train_dir, gesture))

    for img in val_images:
        shutil.copy(os.path.join(gesture_path, img), os.path.join(val_dir, gesture))

    for img in test_images:
        shutil.copy(os.path.join(gesture_path, img), os.path.join(test_dir, gesture))
