import os
import cv2
import imutils
import numpy as np
from tf_keras.models import load_model
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.preprocessing.image import img_to_array

# Global variables
bg = None

def run_avg(image, accumWeight):
    """
    Computes the running average over the background frame.
    """
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    """
    Segments the region of hand in the image.
    """
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        segmented = max(contours, key=cv2.contourArea)
        return thresholded, segmented

def preprocess_and_match_dataset(image):
    """
    Converts the segmented hand to match the dataset's outline style.
    """
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)

    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

    # Apply adaptive thresholding to mimic dataset contrast
    adaptive_threshold = cv2.adaptiveThreshold(
        blurred_image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply Canny edge detection for inner and outer edge detection
    edges = cv2.Canny(adaptive_threshold, 30, 150)

    # Refine edges using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Combine inner features and outlines for a similar appearance to the dataset
    combined = cv2.bitwise_and(adaptive_threshold, cleaned_edges)

    return combined

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
        model = load_model("hand_gesture_recognition_hierarchical.h5")
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
    gestures = ['5', '1', '2', '7', '6', '3', '9', '4', '8', 'v', 'c']
    return gestures[predicted_class] if 0 <= predicted_class < len(gestures) else "Unknown"

def set_background(camera, width=700, height=700):
    """
    Capture the plain background when the camera is first opened.
    """
    ret, frame = camera.read()
    if not ret:
        print("Error: Unable to access the camera.")
        return None
    frame = imutils.resize(frame, width=width, height=height)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    model = load_model_weights()
    recognized_gestures = []
    last_gesture = ""
    current_gesture = ""
    k = 0

    print("[STATUS] Capturing plain background...")
    plain_background = set_background(camera)
    if plain_background is None:
        camera.release()
        cv2.destroyAllWindows()
        exit()

    print("[STATUS] Plain background captured successfully.")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] Please wait! Calibrating...")
            elif num_frames == 29:
                print("[STATUS] Calibration successful...")
        else:
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand
                dataset_style_image = preprocess_and_match_dataset(thresholded)

                cv2.imshow("Dataset Style Hand Outline", dataset_style_image)

                if k % (fps // 6) == 0:
                    cv2.imwrite('Temp.png', dataset_style_image)
                    gesture = predict_gesture(model)

                    if gesture != last_gesture and gesture != "Blank":
                        recognized_gestures.append(gesture)
                        last_gesture = gesture
                        current_gesture = gesture  # Update the current gesture

                # Display the current gesture in the live cam feed
                cv2.putText(clone, f"Current Sign: {current_gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                keypress = cv2.waitKey(1) & 0xFF

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Video Feed", clone)
        num_frames += 1
        k += 1

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break

    print("Recognized Gestures:")
    print(" ".join(recognized_gestures))
    camera.release()
    cv2.destroyAllWindows()
