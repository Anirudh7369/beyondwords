import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

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
    Combines edges and depth-like shading for a 3D effect.    """
  # Calculate gradient for shading effect
    gradient_x = cv2.Sobel(gray_frame, cv2.CV_16F, 0, 0, ksize=5)
    gradient_y = cv2.Sobel(gray_frame, cv2.CV_16F, 0, 0, ksize=5)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Combine edges with the gradient shading
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    gradient_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)

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

def segment_hand(frame, mode="3D"):
    """
    Segments the hand using MediaPipe landmarks and generates either:
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
    edges = cv2.adaptiveThreshold(filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Refine edges to remove noise
    refined_edges = refine_edges(edges)

    if mode == "3D":
        # Generate 3D-like effect
        return create_3d_effect(refined_edges, gray_frame)
    elif mode == "BlackBackground":
        # Generate black background with white outlines
        return create_black_background_with_white_outlines(frame)

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Select the mode: "3D" or "BlackBackground"
    mode = "BlackBackground"  # Change to "3D" for 3D effect
    segmented_hand = segment_hand(frame, mode=mode)

    # Resize the output for better visualization
    segmented_hand_small = cv2.resize(segmented_hand, (440, 380))

    # Display the result
    window_title = "3D Hand Outlines with Reduced Noise" if mode == "3D" else "Black Background with White Outlines"
    cv2.imshow(window_title, segmented_hand_small)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
