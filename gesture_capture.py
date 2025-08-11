import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Define the dataset path
dataset_path = r"C:\Users\ASUS\PycharmProjects\BeyondWords\Final_Dataset"  # Change to your desired dataset path

def ensure_directory_exists(path):
    """
    Ensures that a directory exists. Creates it if it doesn't.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def enhance_contrast(image):
    """
    Enhances the contrast of the image using CLAHE.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def refine_edges(edges):
    """
    Refines edges by removing noise and smoothing outlines.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges

def create_black_background_with_white_outlines(frame):
    """
    Converts the frame to a black background with white hand outlines.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to extract outlines
    edges = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Invert the edges to make the background black and outlines white
    inverted_edges = cv2.bitwise_not(edges)

    return inverted_edges

def segment_hand(frame, mode="BlackBackground"):
    """
    Segments the hand using Mediapipe landmarks and generates threshold images.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better edge detection
    enhanced_gray = enhance_contrast(gray_frame)

    # Apply bilateral filtering to reduce noise while preserving edges
    filtered_gray = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply adaptive thresholding for sharp outlines
    edges = cv2.adaptiveThreshold(filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Refine edges to remove noise
    refined_edges = refine_edges(edges)

    if mode == "BlackBackground":
        return create_black_background_with_white_outlines(frame)

    return None  # Default fallback

# Open the webcam
cap = cv2.VideoCapture(0)

# Counter for images saved per folder
image_count = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Process the frame to apply the black background with white outlines
    processed_frame = segment_hand(frame, mode="BlackBackground")

    # Resize the output for better visualization
    processed_frame_small = cv2.resize(processed_frame, (640, 480))

    # Display the processed frame
    cv2.imshow("Black Background with White Outlines", processed_frame_small)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit the loop with 'q'
        print("Exiting...")
        break

    elif chr(key).isalnum():  # Check if the key pressed is alphanumeric6
        folder_name = chr(key).upper()  # Use uppercase for consistency
        save_path = os.path.join(dataset_path, folder_name)

        # Ensure the directory for this key exists
        ensure_directory_exists(save_path)

        # Initialize or increment the image count for this folder
        if folder_name not in image_count:
            image_count[folder_name] = 0
        image_count[folder_name] += 1

        # Save the processed image
        image_filename = f"{folder_name}_{image_count[folder_name]:04d}.jpg"
        full_save_path = os.path.join(save_path, image_filename)
        cv2.imwrite(full_save_path, processed_frame)

        print(f"Saved: {full_save_path}")

# Release resources
cap.release()
cv2.destroyAllWindows()
