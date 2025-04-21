import cv2
import mediapipe as mp
import math

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is raised by comparing tip and PIP joint positions"""
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y  # Finger is up if tip is above PIP joint

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process both hands and pose
    hand_result = hands.process(rgb)
    pose_result = pose.process(rgb)
    
    hand_raised = False
    fingers_up = False
    arm_raised = False

    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Check if any finger is up (thumb excluded as it's less reliable)
            fingers_up = any([
                is_finger_up(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                is_finger_up(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                is_finger_up(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                is_finger_up(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
            ])
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pose_result.pose_landmarks:
        # Get relevant landmarks
        landmarks = pose_result.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Check if either arm is raised (elbow above shoulder)
        arm_raised = (left_elbow.y < left_shoulder.y) or (right_elbow.y < right_shoulder.y)
        
        mp_drawing.draw_landmarks(
            frame, 
            pose_result.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
    
    # Final decision: hand is raised only if both arm is raised AND fingers are up
    hand_raised = arm_raised and fingers_up

    # Display result
    status_msg = '✋ Hand Raised!' if hand_raised else 'Waiting for hand raise...'
    color = (0, 255, 0) if hand_raised else (0, 0, 255)
    cv2.putText(frame, status_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Additional info
    cv2.putText(frame, f'Arm Raised: {"Yes" if arm_raised else "No"}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Fingers Up: {"Yes" if fingers_up else "No"}', (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Advanced Hand Raise Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()