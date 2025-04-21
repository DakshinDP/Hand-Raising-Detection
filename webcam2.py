import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Colors for different people
    person_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for pose landmarks
        results = pose.process(frame_rgb)
        total_hands_raised = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            person_id = 0  # Currently only one person
            
            # Get key positions
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Check for raised hands - wrist above elbow and elbow above shoulder
            left_hand_raised = False
            right_hand_raised = False

            # Left hand check
            if (left_wrist.visibility > 0.5 and left_elbow.visibility > 0.5 and 
                left_wrist.y < left_elbow.y and left_elbow.y < left_shoulder.y):
                left_hand_raised = True
                total_hands_raised += 1

            # Right hand check
            if (right_wrist.visibility > 0.5 and right_elbow.visibility > 0.5 and 
                right_wrist.y < right_elbow.y and right_elbow.y < right_shoulder.y):
                right_hand_raised = True
                total_hands_raised += 1

            # Draw landmarks with person-specific color (FIXED THIS PART)
            mp_drawing.draw_landmarks(
    frame,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=person_colors[person_id % len(person_colors)]),
    connection_drawing_spec=mp_drawing.DrawingSpec(color=person_colors[person_id % len(person_colors)])
)

            # Display hand raise status for this person
            status_text = f"Person {person_id+1}: "
            if left_hand_raised and right_hand_raised:
                status_text += "Both hands raised"
            elif left_hand_raised:
                status_text += "Left hand raised"
            elif right_hand_raised:
                status_text += "Right hand raised"
            else:
                status_text += "No hands raised"
            
            cv2.putText(frame, status_text, (10, 30 + person_id * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, person_colors[person_id % len(person_colors)], 2)

        # Display total count
        cv2.putText(frame, f'Total Hands Raised: {total_hands_raised}', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Multi-Person Hand Raise Detector', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
