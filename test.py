import cv2
import imutils
import numpy as np
from tf_keras.models import load_model
from tf_keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tf_keras.preprocessing.image import img_to_array
from PIL import Image

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
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        segmented = max(contours, key=cv2.contourArea)
        return thresholded, segmented

def preprocess_image(image_path):
    """
    Preprocesses the image for model prediction.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MobileNetV2

    # Resize image to 224x224 as expected by MobileNetV2
    image = cv2.resize(image, (224, 224))

    # Convert image to array and preprocess it
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Apply MobileNetV2 preprocessing
    return image

def load_model_weights():
    """
    Loads the trained MobileNetV2 model for gesture recognition.
    """
    try:
        model = load_model("hand_gesture_recognition_mobilenetv2.h5")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_gesture(model):
    """
    Predicts the gesture using the trained model.
    """
    processed_image = preprocess_image('Temp.png')  # Assuming Temp.png is the captured image
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    gestures = ['5', '1', '2', '7', 'u', 'w', 'v', 'a', 'b', 'd', 'e']
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
    k = 0

    # Set the plain background at the start (green screen-like effect)
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
            # Subtract the plain background from the current frame
            hand = segment(gray)
            if hand is not None:
                thresholded, segmented = hand

                # Smooth the thresholded image to avoid distortion
                kernel = np.ones((3, 3), np.uint8)
                thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

                # Draw contours of the hand region
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thresholded", thresholded)

                if k % (fps // 6) == 0:
                    cv2.imwrite('Temp.png', thresholded)  # Save the thresholded image
                    gesture = predict_gesture(model)

                    if gesture != last_gesture and gesture != "Blank":
                        recognized_gestures.append(gesture)
                        last_gesture = gesture

                    cv2.putText(clone, gesture, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Green screen effect - Add the detected hand to the plain background
                mask = cv2.bitwise_not(thresholded)

                # Ensure the mask is the same size as the plain background
                mask_resized = cv2.resize(mask, (plain_background.shape[1], plain_background.shape[0]))

                # Ensure that mask is in the correct 8-bit unsigned format
                mask_resized = mask_resized.astype(np.uint8)

                # Resize the plain background to match the frame size
                plain_background_resized = cv2.resize(plain_background, (clone.shape[1], clone.shape[0]))

                # Create the hand area with the mask
                hand_area = cv2.bitwise_and(plain_background_resized, plain_background_resized, mask=mask_resized)

                # Resize clone to ensure it has 3 channels (if it is grayscale, convert it to BGR)
                if len(clone.shape) == 2:  # Grayscale image
                    clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)

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
