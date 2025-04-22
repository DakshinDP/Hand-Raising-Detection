import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Load video file
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Replace with your video path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip if needed, and convert to RGB
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb)
    hand_raised = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist_y = hand_landmarks.landmark[0].y
            if wrist_y < 0.5:  # Normalized y < 0.5 → wrist above halfway → raised
                hand_raised = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display result
    msg = '✋ Hand Raised Detected!' if hand_raised else 'No Hand Raised'
    color = (0, 255, 0) if hand_raised else (0, 0, 255)
    cv2.putText(frame, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('Video Hand Raise Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
