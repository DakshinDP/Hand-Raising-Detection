import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Load image
image_path = 'sat.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    print(f"Error: Unable to load image from path: {image_path}")
    exit()

# Resize for consistent display
image = cv2.resize(image, (640, 480))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image for pose landmarks
results = pose.process(image_rgb)
hand_raised_count = 0

if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Iterate through the detected landmarks (landmark group of 33)
    for i in range(0, len(landmarks), 33):  # Step by 33 landmarks per person
        # Get shoulder and wrist positions for both arms
        left_shoulder = landmarks[i + mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[i + mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[i + mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[i + mp_pose.PoseLandmark.RIGHT_WRIST]

        # Check if left wrist is above left shoulder (indicating raised left hand)
        if left_wrist.visibility > 0.5 and left_wrist.y < left_shoulder.y:
            hand_raised_count += 1
            cv2.putText(image, '👋 Left Hand Raised', (30, 100 + (i * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Check if right wrist is above right shoulder (indicating raised right hand)
        if right_wrist.visibility > 0.5 and right_wrist.y < right_shoulder.y:
            hand_raised_count += 1
            cv2.putText(image, '👋 Right Hand Raised', (30, 140 + (i * 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Draw landmarks on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display total count of raised hands
cv2.putText(image, f'Total Hands Raised: {hand_raised_count}', (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Show the image with raised hands marked
cv2.imshow('Hand Raise Detector (Image)', image)

# Wait for any key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
