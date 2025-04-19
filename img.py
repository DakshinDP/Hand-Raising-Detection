import cv2
import mediapipe as mp

# Initialize mediapipe hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

# Read the image
image_path = 'group.jpg'  # 🔁 Replace with your image filename
image = cv2.imread(image_path)

if image is None:
    print("Image not found.")
    exit()
image = cv2.resize(image, (640, 480))
# Flip and convert to RGB
image = cv2.flip(image, 1)
h, w, _ = image.shape
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

result = hands.process(rgb)
hand_raised = False

if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        wrist_y = hand_landmarks.landmark[0].y
        wrist_pixel_y = wrist_y * h

        # Simple logic: wrist above middle of image => raised hand
        if wrist_pixel_y < h / 2:
            hand_raised = True

        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Display result
if hand_raised:
    cv2.putText(image, '✋ Hand Raised Detected!', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
else:
    cv2.putText(image, 'No Hand Raised', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

cv2.imshow('Hand Raise Detection - Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
