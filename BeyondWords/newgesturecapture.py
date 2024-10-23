import cv2
import os
from datetime import datetime

# Define paths for the directories
base_path = "dataset"
categories = ['hello', 'help', 'namaste']
paths = {category: os.path.join(base_path, category) for category in categories}

# Create directories if they don't exist
for path in paths.values():
    os.makedirs(path, exist_ok=True)

# Start capturing images
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the captured frame
    cv2.imshow("Capture Image", frame)

    # Capture image based on keypress
    key = cv2.waitKey(1)
    if key == ord('h'):  # 'h' key for hello
        img_name = f"{paths['hello']}/hello_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")

    elif key == ord('e'):  # 'e' key for help
        img_name = f"{paths['help']}/help_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")

    elif key == ord('n'):  # 'n' key for namaste
        img_name = f"{paths['namaste']}/namaste_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Captured {img_name}")

    elif key == ord('q'):  # 'q' key to quit
        print("Exiting...")
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
