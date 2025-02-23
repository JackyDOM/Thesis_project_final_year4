import cv2
import mediapipe as mp
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import pyautogui

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    thumb_tip = 4
    fingers_up = 0

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers_up += 1

    if hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 2].x:
        fingers_up += 1  # Thumb considered "up"

    return fingers_up

# Start Selenium WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get('https://poki.com/en/g/subway-surfers')
time.sleep(10)  # Wait for the game to load

cap = cv2.VideoCapture(0)

game_started = False  # Flag to track if the game has started

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_landmarks)

            if fingers == 5 and not game_started:
                print("5 fingers detected: Starting Game!")
                pyautogui.press('space')  # Simulate space key to start the game
                game_started = True  # Prevent multiple starts
                time.sleep(3)  # Short delay before movement starts
            
            if game_started:
                if fingers == 1:
                    # print("1 finger detected: Moving Right!")
                    pyautogui.press('right')
                    time.sleep(0.5)  # Reduce repeat input rate
                elif fingers == 2:
                    # print("2 fingers detected: Moving Left!")
                    pyautogui.press('left')
                    time.sleep(0.5)
                elif fingers == 3:
                    # print("3 fingers detected: Jump!")
                    pyautogui.press('up')
                    time.sleep(0.5)
                elif fingers == 4:
                    # print("4 fingers detected: Roll!")
                    pyautogui.press('down')
                    time.sleep(0.5)

    cv2.imshow("Hand Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
driver.quit()
