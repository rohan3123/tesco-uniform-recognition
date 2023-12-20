import cv2
import numpy as np

# Load pre-trained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images of Tesco workers for recognition
tesco_worker_images = {
    'Employee1': cv2.imread('employees/1.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee2': cv2.imread('employees/2.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee3': cv2.imread('employees/3.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee4': cv2.imread('employees/4.jpg', cv2.IMREAD_GRAYSCALE),
    'Employee5': cv2.imread('employees/5.jpg', cv2.IMREAD_GRAYSCALE),
}

# Function to recognize Tesco worker based on uniform
def recognize_tesco_worker(frame):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = gray[y:y + h, x:x + w]

        # Iterate through Tesco worker images for template matching
        for name, template in tesco_worker_images.items():
            # Perform template matching
            result = cv2.matchTemplate(face_roi, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)

            # Define a confidence threshold (adjust as needed)
            threshold = 0.7

            # If confidence is above threshold, consider it a match
            if confidence > threshold:
                cv2.putText(frame, f'Tesco Worker: {name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Capture video from the default camera (you can modify this to use a specific video file)
cap = cv2.VideoCapture(0)

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Call the recognition function
    frame_with_recognition = recognize_tesco_worker(frame)

    # Display the resulting frame
    cv2.imshow('Tesco Worker Recognition', frame_with_recognition)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
