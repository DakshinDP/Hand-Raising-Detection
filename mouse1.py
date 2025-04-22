import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math
# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Click detection parameters
CLICK_DISTANCE_THRESHOLD = 0.05  # Distance between thumb and index finger tips for click
RIGHT_CLICK_HOLD_TIME = 0.5      # Time in seconds to hold for right click

# Variables for click tracking
click_start_time = 0
click_active = False
last_click_time = 0
click_cooldown = 0.2  # seconds between clicks

# Smoothing parameters (for cursor movement)
smoothing_factor = 0.5
prev_x, prev_y = 0, 0

# Load video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Webcam capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def map_coordinates(x, y, frame_w, frame_h):
    """Map hand coordinates to screen coordinates"""
    # Flip x coordinate (mirror effect)
    x = 1 - x
    # Scale to screen dimensions
    screen_x = np.interp(x, [0, 1], [0, screen_w])
    screen_y = np.interp(y, [0, 1], [0, screen_h])
    return screen_x, screen_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    hand_result = hands.process(rgb)
    
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Get landmarks for index finger and thumb
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Map index finger position to screen coordinates
            screen_x, screen_y = map_coordinates(index_tip.x, index_tip.y, frame_w, frame_h)
            
            # Apply smoothing
            smooth_x = prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor
            smooth_y = prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor
            prev_x, prev_y = smooth_x, smooth_y
            
            # Move mouse pointer
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # Calculate distance between thumb and index finger
            thumb_index_dist = distance(thumb_tip, index_tip)
            
            current_time = time.time()
            
            # Check for click conditions
            if thumb_index_dist < CLICK_DISTANCE_THRESHOLD:
                if not click_active:
                    click_active = True
                    click_start_time = current_time
                else:
                    # Check if we should trigger a right click
                    hold_duration = current_time - click_start_time
                    if hold_duration >= RIGHT_CLICK_HOLD_TIME and (current_time - last_click_time) > click_cooldown:
                        pyautogui.rightClick()
                        last_click_time = current_time
                        click_active = False
            else:
                if click_active:
                    # Check if we should trigger a left click
                    hold_duration = current_time - click_start_time
                    if hold_duration < RIGHT_CLICK_HOLD_TIME and (current_time - last_click_time) > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
                click_active = False
            
            # Draw landmarks (optional, can be disabled when minimized)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Visual feedback for click state
            if click_active:
                hold_duration = time.time() - click_start_time
                if hold_duration >= RIGHT_CLICK_HOLD_TIME:
                    cv2.putText(frame, "RIGHT CLICK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "LEFT CLICK", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame (can minimize this window)
    cv2.imshow('Hand Gesture Mouse', frame)
    
    # Exit on ESC or 'q'
    key = cv2.waitKey(1)
    if key & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()