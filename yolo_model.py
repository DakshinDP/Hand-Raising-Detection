from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model
model = YOLO("yolo11n.pt")  # Replace with your trained model path

# Load video (could also use 0 for webcam)
cap = cv2.VideoCapture("hr2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed if needed
    frame_resized = cv2.resize(frame, (640, 480))

    # Run detection
    results = model(frame_resized, verbose=False)[0]

    count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.5:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"Hand Raised {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display count
    cv2.putText(frame_resized, f"Total Hands Raised: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Classroom Hand Raise Detection", frame_resized)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
