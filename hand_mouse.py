import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Get your screen size
screen_width, screen_height = pyautogui.size()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start camera (1 for DroidCam)
cap = cv2.VideoCapture(1)

click_threshold = 30  # Distance between thumb and index to trigger click
click_delay = 0.3     # Delay between clicks
last_click_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Move mouse with index finger
            index_id, x, y = lm_list[8]
            cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

            # Scale and move cursor
            screen_x = screen_width * x / w
            screen_y = screen_height * y / h
            pyautogui.moveTo(screen_x, screen_y)

            # Detect click: distance between thumb (4) and index (8)
            thumb_id, tx, ty = lm_list[4]
            distance = math.hypot(tx - x, ty - y)
            current_time = time.time()

            if distance < click_threshold and (current_time - last_click_time) > click_delay:
                pyautogui.click()
                last_click_time = current_time
                cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
