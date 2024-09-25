import cv2

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is the default camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame (e.g., applying transformations)
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Example transformation: Convert to grayscale

    # Display the original frame
    cv2.imshow('Original Frame', frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

