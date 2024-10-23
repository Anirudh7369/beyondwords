import cv2
import numpy as np
import tensorflow as tf
from tf_keras.models import load_model
import pyttsx3
import mediapipe as mp

# Load the trained model
model = load_model('my_model.h5')  # Ensure the model path is correct

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define sign classes including 'No Sign'
sign_classes = ['no_sign', 'hello', 'help', 'namaste']

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Adjusted detection confidence
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam for predictions
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural visualization
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe uses RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            img_height, img_width, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img_width)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img_width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img_height)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img_height)

            # Extract the hand region of interest (ROI)
            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size > 0:
                hand_roi_resized = cv2.resize(hand_roi, (64, 64))
                hand_roi_rgb = cv2.cvtColor(hand_roi_resized, cv2.COLOR_BGR2RGB)
                hand_roi_normalized = hand_roi_rgb.astype('float32') / 255.0
                hand_roi_input = hand_roi_normalized.reshape(1, 64, 64, 3)

                # Make prediction
                predictions = model.predict(hand_roi_input)

                predicted_index = np.argmax(predictions)
                predicted_class = sign_classes[predicted_index]
                confidence = predictions[0][predicted_index]

                # Adjust confidence threshold for outputs
                if confidence > 0.3:
                    print(f'Final Predicted Sign: {predicted_class} (Confidence: {confidence:.2f})')
                    engine.say(predicted_class)
                    engine.runAndWait()
                    cv2.putText(frame, f'{predicted_class} ({confidence:.2f})', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                else:
                    cv2.putText(frame, 'No Sign Detected', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        cv2.putText(frame, 'No Hand Detected', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the Esc key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
