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

# Control parameters
CLICK_DISTANCE_THRESHOLD = 0.05
RIGHT_CLICK_HOLD_TIME = 0.5
MOVEMENT_SCALE = 1.8  # Sensitivity multiplier

# Variables for click tracking
click_start_time = 0
click_active = False
last_click_time = 0

# Smoothing parameters
smoothing_factor = 0.3
prev_x, prev_y = screen_w/2, screen_h/2  # Start at center

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def map_coordinates(x, y, frame_w, frame_h):
    """Natural movement mapping (non-inverted) with flip compensation"""
    # Flip the x-coordinate to match the displayed frame
    flipped_x = 1 - x
    
    # Apply sensitivity scaling from center
    scaled_x = 0.5 + (flipped_x - 0.5) * MOVEMENT_SCALE
    scaled_y = 0.5 + (y - 0.5) * MOVEMENT_SCALE
    
    # Clamp to valid range
    scaled_x = max(0, min(1, scaled_x))
    scaled_y = max(0, min(1, scaled_y))
    
    # Map to screen coordinates
    screen_x = np.interp(scaled_x, [0, 1], [0, screen_w])
    screen_y = np.interp(scaled_y, [0, 1], [0, screen_h])
    
    return screen_x, screen_y

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror effect (visual only)
    flipped_frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # Convert original frame to RGB for processing (not the flipped one)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands on original frame
    hand_result = hands.process(rgb)
    
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            # Get landmarks from original frame
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Map to screen coordinates with flip compensation
            screen_x, screen_y = map_coordinates(index_tip.x, index_tip.y, frame_w, frame_h)
            
            # Apply smoothing
            smooth_x = prev_x * (1 - smoothing_factor) + screen_x * smoothing_factor
            smooth_y = prev_y * (1 - smoothing_factor) + screen_y * smoothing_factor
            prev_x, prev_y = smooth_x, smooth_y
            
            # Move mouse pointer (natural direction)
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # Click detection
            current_time = time.time()
            thumb_dist = distance(thumb_tip, index_tip)
            
            if thumb_dist < CLICK_DISTANCE_THRESHOLD:
                if not click_active:
                    click_active = True
                    click_start_time = current_time
                elif current_time - click_start_time >= RIGHT_CLICK_HOLD_TIME:
                    if current_time - last_click_time > 0.3:
                        pyautogui.rightClick()
                        last_click_time = current_time
                        click_active = False
            else:
                if click_active:
                    if current_time - last_click_time > 0.1:
                        pyautogui.click()
                        last_click_time = current_time
                click_active = False
            
            # Draw landmarks on the flipped frame for display
            mp_drawing.draw_landmarks(flipped_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if click_active:
                status = "RIGHT CLICK" if (current_time - click_start_time) >= RIGHT_CLICK_HOLD_TIME else "LEFT CLICK"
                color = (0, 0, 255) if "RIGHT" in status else (0, 255, 0)
                cv2.putText(flipped_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display the flipped frame
    cv2.imshow('Hand Mouse Control', flipped_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()