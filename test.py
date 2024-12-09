import cv2
import numpy as np
import imutils
from tf_keras.models import load_model
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.preprocessing.image import img_to_array

# Global variables
bg = None

# Functions from the Threshold Code
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

def create_3d_effect(edges, gray_frame):
    """
    Combines edges and depth-like shading for a 3D effect.
    """
    gradient_x = cv2.Sobel(gray_frame, cv2.CV_16F, 0, 1, ksize=5)
    gradient_y = cv2.Sobel(gray_frame, cv2.CV_16F, 1, 0, ksize=5)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine edges with the gradient shading
    gradient_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Blend edges with gradient shading
    combined = cv2.addWeighted(edges_colored, 0.7, gradient_colored, 0.3, 0)
    return combined

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
    Segments the hand using Mediapipe landmarks and generates either:
    - "3D": A 3D-like effect for the outlines.
    - "BlackBackground": A black background with white hand outlines.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better edge detection
    enhanced_gray = enhance_contrast(gray_frame)

    # Apply bilateral filtering to reduce noise while preserving edges
    filtered_gray = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply adaptive thresholding for sharp outlines
    edges = cv2.adaptiveThreshold(filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Refine edges to remove noise
    refined_edges = refine_edges(edges)

    if mode == "3D":
        # Generate 3D-like effect
        return create_3d_effect(refined_edges, gray_frame)
    elif mode == "BlackBackground":
        # Generate black background with white outlines
        return create_black_background_with_white_outlines(frame)

# Test Code Logic
def preprocess_image(image_path):
    """
    Preprocesses the image for model prediction.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def load_model_weights():
    """
    Loads the trained MobileNetV2 model for gesture recognition.
    """
    try:
        model = load_model('Final_fine_tuned_model.h5')
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_gesture(model):
    """
    Predicts the gesture using the trained model.
    """
    processed_image = preprocess_image('Temp.png')
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    gestures = {
        0: '1', 1: '2', 2: '3', 3: '4', 4: '5',
        5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D', 13: 'V'
    }
    return gestures.get(predicted_class, "Unknown")

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    model = load_model_weights()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Apply the same Threshold Code logic
        mode = "BlackBackground"  # Change to "3D" for 3D effect
        threshold_output = segment_hand(frame, mode=mode)

        # Display the threshold output
        cv2.imshow("Threshold Output", threshold_output)

        # Save the thresholded frame for model prediction
        cv2.imwrite('Temp.png', threshold_output)

        # Predict the gesture
        gesture = predict_gesture(model)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the video feed with gesture overlay
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
