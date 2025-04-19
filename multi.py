import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture('hr1.mp4') 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent display
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame_rgb)
    hand_raised_count = 0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Loop through all people (if available)
        # We will track multiple people based on their wrist's position.
        for i in range(0, len(landmarks), 33):  # Step by 33 landmarks per person
            left_wrist = landmarks[i + mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[i + mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[i + mp_pose.PoseLandmark.NOSE]

            # Check if either wrist is higher than the nose (basic raised hand logic)
            if left_wrist.visibility > 0.5 and left_wrist.y < nose.y:
                hand_raised_count += 1
                cv2.putText(frame, '👋 Left Hand Raised', (30, 100 + (i * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if right_wrist.visibility > 0.5 and right_wrist.y < nose.y:
                hand_raised_count += 1
                cv2.putText(frame, '👋 Right Hand Raised', (30, 140 + (i * 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw skeleton for all detected people
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display total count of hands raised
    cv2.putText(frame, f'Total Hands Raised: {hand_raised_count}', (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('Hand Raise Detector (Multiple People)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
