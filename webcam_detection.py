# Install necessary libraries


# Import the libraries
from ultralytics import YOLO
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the YOLOv8 model (pre-trained for detecting hands or general objects)
model = YOLO('/content/yolo11n.pt')  # You can choose different YOLO models (like 'yolov8m.pt' or 'yolov8l.pt')

# Open the webcam (0 for default camera)
cap = cv2.VideoCapture(0)

# Set the video width and height
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Start video capture loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break
    
    # Run hand detection using the YOLO model
    results = model(frame)

    # Process the detections
    for result in results:
        boxes = result.boxes.xyxy  # Get the bounding boxes
        confidences = result.boxes.conf  # Get confidence scores
        class_ids = result.boxes.cls  # Get class IDs

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  # Confidence threshold for detecting objects (hands in this case)
                x1, y1, x2, y2 = map(int, box)  # Get box coordinates
                label = f'{model.names[int(class_id)]}: {conf:.2f}'  # Class label and confidence
                color = (0, 255, 0)  # Green color for the bounding box

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame with bounding boxes
    cv2_imshow(frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
