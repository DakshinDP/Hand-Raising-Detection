import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Test dataset: (image_path, actual_raised_hands)
test_data = [
    ("img1.png", 2),
    ("img2.jpg", 1),
    ("img3.png", 3),
    ("img4.jpg", 0)
]

correct_predictions = 0

for image_path, ground_truth_count in test_data:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading {image_path}")
        continue

    image = cv2.resize(image, (640, 480))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image_rgb)
    hand_raised_count = 0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        if left_wrist.visibility > 0.5 and left_wrist.y < left_shoulder.y:
            hand_raised_count += 1
        if right_wrist.visibility > 0.5 and right_wrist.y < right_shoulder.y:
            hand_raised_count += 1

    print(f"{image_path}: Detected={hand_raised_count}, Actual={ground_truth_count}")

    if hand_raised_count == ground_truth_count:
        correct_predictions += 1

# Accuracy calculation
accuracy = (correct_predictions / len(test_data)) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")
