import cv2
import numpy as np
import imutils
from collections import defaultdict
from tf_keras.models import load_model
from tf_keras.applications.mobilenet_v2 import preprocess_input
from tf_keras.preprocessing.image import img_to_array

bg = None
predictions_count = defaultdict(int)  
word = ""  

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def refine_edges(edges):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges

def create_3d_effect(edges, gray_frame):
    gradient_x = cv2.Sobel(gray_frame, cv2.CV_16F, 0, 1, ksize=5)
    gradient_y = cv2.Sobel(gray_frame, cv2.CV_16F, 1, 0, ksize=5)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gradient_colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_JET)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = cv2.addWeighted(edges_colored, 0.7, gradient_colored, 0.3, 0)
    return combined

def create_black_background_with_white_outlines(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_edges = cv2.bitwise_not(edges)
    return inverted_edges

def segment_hand(frame, mode="BlackBackground"):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = enhance_contrast(gray_frame)
    filtered_gray = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
    edges = cv2.adaptiveThreshold(filtered_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    refined_edges = refine_edges(edges)
    if mode == "3D":
        return create_3d_effect(refined_edges, gray_frame)
    elif mode == "BlackBackground":
        return create_black_background_with_white_outlines(frame)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def load_model_weights():
    try:
        model = load_model('Final_fine_tuned_model.h5')
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_gesture(model):
    
    processed_image = preprocess_image('Temp.png')
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    gestures = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'Allah', 12: 'B', 13: 'Blank',
        14: 'C', 15: 'D', 16: 'E', 17: 'F', 18: 'G', 19: 'H', 20: 'I', 21: 'J', 22: 'K',
        23: 'L', 24: 'M', 25: 'N', 26: 'O', 27: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T',
        33: 'U', 34: 'V', 35: 'W', 36: 'X', 38: 'Y', 39: 'Z'
    }
    return gestures.get(predicted_class, "Unknown")  # Default to "Blank" if not found

def update_word(prediction):
    global predictions_count, word

    if prediction == "Blank":
        return  # Ignore "Blank" predictions

    predictions_count[prediction] += 1

    if predictions_count[prediction] > 50:
        max_count = max(predictions_count.values())
        second_max_count = sorted(predictions_count.values(), reverse=True)[1] if len(predictions_count) > 1 else 0

        if max_count - second_max_count > 20:
            word += prediction
            predictions_count.clear()
        else:
            predictions_count.clear()

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    model = load_model_weights()

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        frame = cv2.flip(frame, 1)
        mode = "BlackBackground"
        threshold_output = segment_hand(frame, mode=mode)
        cv2.imshow("Threshold Output", threshold_output)
        cv2.imwrite('Temp.png', threshold_output)

        gesture = predict_gesture(model)
        update_word(gesture)

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Final Word: {word}")
    camera.release()
    cv2.destroyAllWindows()
