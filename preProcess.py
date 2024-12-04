import cv2
import numpy as np
import os
import random


def augment_image(image):
    """Perform augmentations: rotate, flip, add noise."""
    # Random rotation
    angle = random.randint(-15, 15)  # Rotate within [-15, 15] degrees
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=0)

    # Random flip
    flip_code = random.choice([-1, 0, 1])  # Horizontal, vertical, or both
    flipped_image = cv2.flip(rotated_image, flip_code)

    # Add random noise
    noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
    noisy_image = cv2.add(flipped_image, noise)

    return noisy_image


def preprocess_binary_images(input_path, output_path, augmentations=5):
    """
    Preprocess and augment binary threshold images.
    Args:
        input_path (str): Path to the input dataset.
        output_path (str): Path to save the preprocessed images.
        augmentations (int): Number of augmented images to generate per input image.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_name in os.listdir(input_path):
        image_path = os.path.join(input_path, image_name)

        # Read the binary image
        binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if binary_image is None:
            print(f"Skipping {image_name}: Not a valid image file.")
            continue

        # Save original image
        original_output_path = os.path.join(output_path, f"original_{image_name}")
        cv2.imwrite(original_output_path, binary_image)

        # Generate augmented images
        for i in range(augmentations):
            augmented_image = augment_image(binary_image)

            # Save the augmented image
            augmented_output_path = os.path.join(output_path, f"aug_{i}_{image_name}")
            cv2.imwrite(augmented_output_path, augmented_image)

        print(f"Processed and saved augmentations for: {image_name}")

    print("All images processed and augmented successfully.")


# User inputs
input_dataset_path = input("Enter the path to the input dataset: ")
output_dataset_path = input("Enter the path to the output dataset: ")
