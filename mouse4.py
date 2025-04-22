import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Control parameters
CLICK_DISTANCE_THRESHOLD = 0.05
RIGHT_CLICK_HOLD_TIME = 0.8
SCROLL_ACTIVATION_DISTANCE = 0.07  # Distance between middle and index to activate scroll
SCROLL_DEACTIVATION_DISTANCE = 0.1  # Distance to deactivate scroll
SCROLL_SENSITIVITY = 30  # Higher = more scroll per movement
MOVEMENT_SCALE = 1.8
CLICK_MOVE_THRESHOLD = 50  # Max pixels mouse can move during click

# Tracking variables
click_state = {'active': False, 'start_time': 0, 'position': (0, 0), 'last_click': 0}
scroll_state = {'active': False, 'last_y': 0, 'last_scroll': 0}
prev_x, prev_y = screen_w/2, screen_h/2  # Start at center

def distance(p1, p2):
    """Calculate distance between two points (works for both landmarks and tuples)"""
    if hasattr(p1, 'x'):  # MediaPipe landmarks
        return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
    else:  # Screen coordinates (tuples)
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def map_coordinates(x, y):
    """Map normalized coordinates to screen with flip compensation"""
    flipped_x = 1 - x  # Flip X for natural movement
    scaled_x = 0.5 + (flipped_x - 0.5) * MOVEMENT_SCALE
    scaled_y = 0.5 + (y - 0.5) * MOVEMENT_SCALE
    scaled_x = max(0, min(1, scaled_x))
    scaled_y = max(0, min(1, scaled_y))
    return np.interp(scaled_x, [0, 1], [0, screen_w]), np.interp(scaled_y, [0, 1], [0, screen_h])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for display only
    display_frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # Process original frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Map coordinates
        target_x, target_y = map_coordinates(index_tip.x, index_tip.y)
        
        # Smooth cursor movement
        smooth_x = prev_x * 0.7 + target_x * 0.3
        smooth_y = prev_y * 0.7 + target_y * 0.3
        prev_x, prev_y = smooth_x, smooth_y
        
        # Check scroll activation
        middle_index_dist = distance(middle_tip, index_tip)
        current_time = time.time()
        
        # Scroll logic
        if middle_index_dist < SCROLL_ACTIVATION_DISTANCE:
            if not scroll_state['active']:
                scroll_state = {'active': True, 'last_y': middle_tip.y, 'last_scroll': current_time}
            else:
                # Calculate scroll amount based on vertical movement
                scroll_dy = (middle_tip.y - scroll_state['last_y']) * SCROLL_SENSITIVITY
                if abs(scroll_dy) > 0.01 and (current_time - scroll_state['last_scroll']) > 0.05:
                    pyautogui.scroll(int(-scroll_dy))  # Convert to integer
                    scroll_state['last_y'] = middle_tip.y
                    scroll_state['last_scroll'] = current_time
            
            # Visual feedback
            cv2.putText(display_frame, "SCROLLING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif scroll_state['active'] and middle_index_dist > SCROLL_DEACTIVATION_DISTANCE:
            scroll_state['active'] = False
        
        # Only process clicks if not scrolling
        if not scroll_state['active']:
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # Click detection
            thumb_dist = distance(thumb_tip, index_tip)
            
            if thumb_dist < CLICK_DISTANCE_THRESHOLD:
                if not click_state['active']:
                    click_state = {'active': True, 'start_time': current_time, 
                                 'position': (smooth_x, smooth_y), 'last_click': current_time}
                elif current_time - click_state['start_time'] >= RIGHT_CLICK_HOLD_TIME:
                    if (current_time - click_state['last_click']) > 0.5:
                        current_pos = (smooth_x, smooth_y)
                        if distance(current_pos, click_state['position']) < CLICK_MOVE_THRESHOLD:
                            pyautogui.rightClick()
                            click_state['last_click'] = current_time
                            click_state['active'] = False
            else:
                if click_state['active']:
                    if (current_time - click_state['last_click']) > 0.2:
                        current_pos = (smooth_x, smooth_y)
                        if distance(current_pos, click_state['position']) < CLICK_MOVE_THRESHOLD:
                            pyautogui.click()
                            click_state['last_click'] = current_time
                click_state['active'] = False
            
            # Click visual feedback
            if click_state['active']:
                status = "RIGHT CLICK" if (current_time - click_state['start_time']) >= RIGHT_CLICK_HOLD_TIME else "LEFT CLICK"
                color = (0, 0, 255) if "RIGHT" in status else (0, 255, 0)
                cv2.putText(display_frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Draw hand landmarks
        mp_drawing.draw_landmarks(display_frame, hand, mp_hands.HAND_CONNECTIONS)
    
    # Display
    cv2.imshow('Hand Mouse Control', display_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()