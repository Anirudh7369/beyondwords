import cv2
import mediapipe as mp
import pyttsx3
import threading
import math

# Initialize MediaPipe Hands, drawing utils, and pyttsx3 engine
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Global flag to manage TTS thread
tts_in_progress = False


# Function to speak text using threading
def speak_text(text):
    global tts_in_progress
    if not tts_in_progress:  # If no other TTS is in progress
        tts_in_progress = True
        thread = threading.Thread(target=tts_thread, args=(text,))
        thread.start()


def tts_thread(text):
    global tts_in_progress
    engine.say(text)
    engine.runAndWait()
    tts_in_progress = False


# Function to calculate Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


# Initialize the previous gesture and time
prev_gesture = None

# Start the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame skipping for performance
frame_skip = 3  # Process every 3rd frame
frame_count = 0

# Hand detection model
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    while cap.isOpened():
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 3rd frame
        if frame_count % frame_skip != 0:
            continue

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect hands
        results = hands.process(image)

        # Convert back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # If hand landmarks are detected
        gesture = None  # Default gesture is None

        if results.multi_hand_landmarks:
            # If two hands are detected
            if len(results.multi_hand_landmarks) == 2:
                hand1 = results.multi_hand_landmarks[0]
                hand2 = results.multi_hand_landmarks[1]

                # Draw hand landmarks for both hands
                mp_drawing.draw_landmarks(image, hand1, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, hand2, mp_hands.HAND_CONNECTIONS)

            # If only one hand is detected
            elif len(results.multi_hand_landmarks) == 1:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmarks for each hand
                landmarks = hand_landmarks.landmark

                # Get landmarks for fingers
                index_tip = landmarks[8]  # Tip of the index finger
                index_lower = landmarks[6]  # Lower joint of the index finger
                middle_tip = landmarks[12]
                middle_lower = landmarks[10]
                ring_tip = landmarks[16]
                ring_lower = landmarks[14]
                pinky_tip = landmarks[20]
                pinky_lower = landmarks[18]
                thumb_tip = landmarks[4]
                thumb_lower = landmarks[3]

                # Condition for open hand (Hello)
                if (index_tip.y < index_lower.y and
                        middle_tip.y < middle_lower.y and
                        ring_tip.y < ring_lower.y and
                        pinky_tip.y < pinky_lower.y):
                    gesture = "Hello"
                    cv2.putText(image, "Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Condition for closed fist (Help)
                elif (index_tip.y > index_lower.y and
                      middle_tip.y > middle_lower.y and
                      ring_tip.y > ring_lower.y and
                      pinky_tip.y > pinky_lower.y):
                    gesture = "Help"
                    cv2.putText(image, "Help", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Recognize numbers 0 to 5
                # Condition for 0 (closed fist)

                # Condition for 1 (index finger up)
                elif (index_tip.y < index_lower.y and
                      middle_tip.y > middle_lower.y and
                      ring_tip.y > ring_lower.y and
                      pinky_tip.y > pinky_lower.y):
                    gesture = "1"
                    cv2.putText(image, "1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Condition for 2 (index and middle fingers up)
                elif (index_tip.y < index_lower.y and
                      middle_tip.y < middle_lower.y and
                      ring_tip.y > ring_lower.y and
                      pinky_tip.y > pinky_lower.y):
                    gesture = "2"
                    cv2.putText(image, "2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # Condition for 3 (index, middle, and ring fingers up)
                elif (index_tip.y < index_lower.y and
                      middle_tip.y < middle_lower.y and
                      ring_tip.y < ring_lower.y and
                      pinky_tip.y > pinky_lower.y):
                    gesture = "3"
                    cv2.putText(image, "3", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Speak the gesture if recognized
        if gesture is not None and gesture != prev_gesture:
            speak_text(gesture)  # Speak the gesture in a separate thread
            prev_gesture = gesture  # Update previous gesture

        # Display the frame
        cv2.imshow('Hand Gesture Recognition', image)

        # Break loop on 'ESC' key press (key code 27)
        if cv2.waitKey(10) & 0xFF == 27:
            break

# Release video capture
cap.release()
cv2.destroyAllWindows()
