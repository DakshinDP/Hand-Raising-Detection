import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,  # Changed to False for video
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,  # Added for video
                        model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose landmarks
        results = pose.process(frame_rgb)
        hand_raised_count = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get shoulder and wrist positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Check for raised hands
            left_hand_raised = False
            right_hand_raised = False

            if left_wrist.visibility > 0.5 and left_wrist.y < left_shoulder.y:
                hand_raised_count += 1
                left_hand_raised = True

            if right_wrist.visibility > 0.5 and right_wrist.y < right_shoulder.y:
                hand_raised_count += 1
                right_hand_raised = True

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Display hand raise status
            if left_hand_raised:
                cv2.putText(frame, 'Left Hand Raised', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if right_hand_raised:
                cv2.putText(frame, 'Right Hand Raised', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display total count
        cv2.putText(frame, f'Hands Raised: {hand_raised_count}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Hand Raise Detector (Webcam)', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
